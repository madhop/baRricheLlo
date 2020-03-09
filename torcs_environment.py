import gym
from gym import spaces
import snakeoil3_gym as snakeoil3
import numpy as np
import pandas as pd
import copy
import compute_state_features as csf
from scipy import spatial
import auto_driver
import os
import time
from utils_torcs import *


class TORCS(gym.Env):

    def __init__(self, reward_function, state_cols, state_space, ref_df, practice_path, gear_change=False,
                 track_length=5783.85, damage_th=4.0, graphic=True, speed_limit=5, verbose=True,
                 collision_penalty=-1000, low_speed_penalty=-1000):

        self.initial_reset = True
        self.gear_change = gear_change
        self.reward_function = reward_function
        self.speed_limit = speed_limit
        self.verbose = verbose
        self.practice_path = practice_path

        self.graphic = graphic
        self.track_length = track_length
        self.damage_th = damage_th
        self.collision_penalty = collision_penalty
        self.low_speed_penalty = low_speed_penalty

        # save variables used for feature extraction
        self.state_cols = state_cols
        self.ref_df = ref_df
        self.tree = spatial.KDTree(list(zip(ref_df['xCarWorld'], ref_df['yCarWorld'])))

        # Create action space
        # order: steer, brake, throttle<, gear>
        if gear_change:
            high = np.array([1., 1., 1., 7])
            low = np.array([-1., 0., 0., 1])
        else:
            high = np.array([1., 1., 1.])
            low = np.array([-1., 0., 0.])

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Create observation space
        self.observation_space = spaces.Box(low=state_space['low'], high=state_space['high'], dtype=np.float32)

        # Create observation variables that comes from make_observations method
        # they are dictionaries containing raw torcs measurements
        self.observation = None
        self.p1_observation = None
        self.p2_observation = None

        # Create current state variable
        self.state = None

        self.initial_run = True

        self.time_step = 0

    def step(self, u, raw=False):
        """

        :param u: (list) action
        :param raw: (bool) True to return torcs observation, False to return the extracted features i.e., state
        :return: next observation
        """
        client = self.client

        # convert thisAction to the actual torcs actionstr
        this_action = self.agent_to_torcs(u)

        # Get the dictionary of torcs action
        action_torcs = client.R.d

        # Update each field of the action dict with the current action

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        # Throttle
        action_torcs['accel'] = this_action['accel']

        # Gear
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            # return 0 to call automatic gear change
            action_torcs['gear'] = 0

        # Brake
        action_torcs['brake'] = this_action['brake']

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        # Compute reward only if the agent is driving
        if raw:
            reward = 0
        else:
            # Reward computation
            # Compute the current state
            current_state = self.observation_to_state(self.observation, self.p1_observation, self.p2_observation,
                                                      self.prev_u)
            current_state_df = np.concatenate([current_state.reshape(1, -1), np.zeros((1, 1))], axis=1)
            # Compute the next state
            next_state = self.observation_to_state(self.make_observation(obs), self.observation, self.p1_observation, u)
            next_state_df = np.concatenate([next_state.reshape(1, -1), np.ones((1, 1)) * obs['trackPos']], axis=1)
            data = pd.DataFrame(data=np.concatenate([current_state_df, next_state_df], axis=0),
                                columns=self.state_cols + ['trackPos'])
            reward = self.reward_function(data)

        # Save u as previous action for the next step
        self.prev_u = u

        # Save the old observations
        self.p2_observation = self.p1_observation
        self.p1_observation = self.observation
        # Make an observation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)

        # Episode termination checks ###
        episode_terminate = False
        # 1) Start line
        # The car passed the start line if the previous observation is before and the current is after it
        if (self.p1_observation['distFromStart'] > self.track_length - 50) and\
                (self.observation['distFromStart'] < 50) and\
                not raw:
            # we passed the start line
            if self.verbose:
                print('Start reached!')
            checkered_flag = True
            episode_terminate = True
        else:
            checkered_flag = False

        # 2) Collision
        damage = self.observation['damage'] - self.p1_observation['damage']
        if  damage > self.damage_th:
            if self.verbose:
                print('Hit wall')
            collision = True
            episode_terminate = True
            reward += self.collision_penalty
        else:
            collision = False

        # 3) Low speed
        if self.observation['speed_x'] <= self.speed_limit:
            if self.verbose:
                print('Low speed')
            episode_terminate = True
            low_speed = True
            reward += self.low_speed_penalty
        else:
            low_speed = False
            
        # 4) Running backward
        # Episode is terminated if the agent runs backward
        if np.cos(self.observation['angle']) < 0:
            episode_terminate = True
            running_backward = True
        else:
            running_backward = False

        # 5) Out of track
        if abs(self.observation['trackPos']) > 1.01:  # Episode is terminated if the car is out of track
            print('out of track')
            episode_terminate = True
            out_of_track = True
            reward += self.collision_penalty
        else:
            out_of_track = False

        # If there is automatic driving episode terminate is False
        if raw:
            episode_terminate = False

        # Send a reset signal
        if client.R.d['meta'] is True:
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        info = {'collision': collision, 'is_success': checkered_flag, 'low_speed': low_speed,
                'running_backward': running_backward, 'out_of_track': out_of_track }
        if self.verbose:
            print('T={} B={} S={} r={} d={}'.format(action_torcs['accel'], action_torcs['brake'], action_torcs['steer'],
                                                    reward, damage))

        if raw:
            # If raw then return current observation otherwise return the state
            return self.get_obs(), reward, episode_terminate, info
        else:
            # Transform raw torcs data to state
            self.state = self.observation_to_state(self.observation, self.p1_observation, self.p2_observation, u)
            return self.get_state(), reward, episode_terminate, info

    def reset(self, relaunch=False):
        # print("Reset")

        self.time_step = 0

        if not self.initial_reset:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            # TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        # Open new UDP in vtorcs
        self.client = snakeoil3.Client(p=3001, vision=False, graphic=self.graphic, practice_path=self.practice_path)
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        ob = client.S.d  # Get the current full-observation from torcs
        # Save observation variables
        self.observation = self.make_observation(ob)
        self.p1_observation = self.observation
        self.p2_observation = self.observation

        self.last_u = None

        self.initial_reset = False

        auto_drive = True
        print('Started auto driving')
        while auto_drive:
            ob_distFromStart = self.observation['distFromStart']
            track_pos = self.observation['trackPos']
            action = auto_driver.get_action(track_pos)
            self.observation, _, done, info = self.step(action, raw=True)

            if ob_distFromStart < 100:
                # Start with agent driver and return the current state
                print('Stopped auto driving')
                return self.observation_to_state(self.observation, self.p1_observation, self.p2_observation, action)
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def get_state(self):
        return self.state

    def reset_torcs(self):
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.graphic:
            command_str = 'torcs -nofuel -nodamage -nolaptime'
        else:
            command_str = 'torcs -r {} -nofuel -nodamage -nolaptime'.format(self.practice_path)

        os.system(command_str + ' &')
        if self.graphic:
            time.sleep(0.5)
            os.system('sh autostart.sh')
            time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[2]})

        if self.brake is True: # brake action is enabled
            torcs_action.update({'brake': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': u[3]})

        return torcs_action

    def make_observation(self, raw_obs):
        return { k: np.array(raw_obs[k], dtype=np.float32) for k in torcs_features}

        """return {'angle' : np.array(raw_obs['angle'], dtype=np.float32),
                'curLapTime' : np.array(raw_obs['curLapTime'], dtype=np.float32),
                'damage' : np.array(raw_obs['damage'], dtype=np.float32),
                'distFromStart' : np.array(raw_obs['distFromStart'], dtype=np.float32),
                'Acceleration_x' : np.array(raw_obs['Acceleration_x'], dtype=np.float32),
                'Acceleration_y' : np.array(raw_obs['Acceleration_y'], dtype=np.float32),
                'Gear' : np.array(raw_obs['gear'], dtype=np.float32),
                'lastLapTime' : np.array(raw_obs['lastLapTime'], dtype=np.float32),
                'rpm' : np.array(raw_obs['rpm'], dtype=np.float32),
                'speed_x' : np.array(raw_obs['speedX'], dtype=np.float32),
                'speed_y' : np.array(raw_obs['speedY'], dtype=np.float32),
                'speed_z' : np.array(raw_obs['speedZ'], dtype=np.float32),
                'x' : np.array(raw_obs['x'], dtype=np.float32),
                'y' : np.array(raw_obs['y'], dtype=np.float32),
                'z' : np.array(raw_obs['z'], dtype=np.float32),
                'roll' : np.array(raw_obs['roll'], dtype=np.float32),
                'pitch' : np.array(raw_obs['pitch'], dtype=np.float32),
                'yaw' : np.array(raw_obs['yaw'], dtype=np.float32),
                'speedGlobalX' : np.array(raw_obs['speedGlobalX'], dtype=np.float32),
                'speedGlobalY' : np.array(raw_obs['speedGlobalY'], dtype=np.float32)}"""

    def observation_to_state(self, ob, p_1, p_2, prev_action):
        """Transform raw observation into state observation with extracted features
        :param ob: (dict) raw torcs features
        :param p_1: (dict) previous observation
        :param p_2: (dict) two times previous observation
        :param prev_action: (list) previous actions

        :return obs: (list) state features in the order of the self.state_cols list
        """
        p = ob
        nn = csf.nn_kdtree(np.array([p['x'], p['y']]), self.tree)
        # check if you are at the end of the lap
        if nn >= self.ref_df.shape[0] - 1:
            nn = self.ref_df.shape[0] - 2
        r = self.ref_df.iloc[nn]
        r1 = self.ref_df.iloc[nn + 1]
        r_1 = self.ref_df.iloc[nn - 1]

        v_actual_module, v_ref_module, v_diff_module, v_diff_of_modules, v_angle = csf.velocity_acceleration(
            np.array([p['speed_x'], p['speed_y']]), np.array([r['speed_x'], r['speed_y']]))
        rel_p, rho, theta = csf.position(np.array([r['xCarWorld'], r['yCarWorld']]),
                                         np.array([r1['xCarWorld'], r1['yCarWorld']]), np.array([p['x'], p['y']]))
        actual_c = csf.curvature(np.array([p['x'], p['y']]), np.array([p_1['x'], p_1['y']]),
                                 np.array([p_2['x'], p_2['y']]))
        ref_c = csf.curvature(np.array([r1['xCarWorld'], r1['yCarWorld']]), np.array([r['xCarWorld'], r['yCarWorld']]),
                              np.array([r_1['xCarWorld'], r_1['yCarWorld']]))
        direction = csf.direction(np.array([p['x'], p['y']]), np.array([p_1['x'], p_1['y']]))

        state_features = {'xCarWorld': p['x'], 'yCarWorld': p['y'], 'nYawBody': p['yaw'], 'nEngine': p['rpm'],
                          'NGear': p['Gear'],
                          'speed_x': p['speed_x'],
                          'positionRho': rho, 'positionTheta': theta, 'positionReferenceX': r['xCarWorld'],
                          'positionReferenceY': r['yCarWorld'],
                          'positionRelativeX': rel_p[0], 'positionRelativeY': rel_p[1], 'referenceCurvature': ref_c,
                          'actualCurvature': actual_c,
                          'actualSpeedModule': v_actual_module, 'speedDifferenceVectorModule': v_diff_module,
                          'speedDifferenceOfModules': v_diff_of_modules,
                          'actualAccelerationX': p['Acceleration_x'], 'actualAccelerationY': p['Acceleration_y'],
                          'referenceAccelerationX': r['Acceleration_x'], 'referenceAccelerationY': r['Acceleration_y'],
                          'accelerationDiffX': r['Acceleration_x'] - p['Acceleration_x'],
                          'accelerationDiffY': r['Acceleration_y'] - p['Acceleration_y'],
                          'direction_x': direction[0],
                          'direction_y': direction[1],
                          'prevaSteerWheel': prev_action[0], 'prevpBrakeF': prev_action[2],
                          'prevrThrottlePedal': prev_action[1]}

        return np.array([state_features[k] for k in self.state_cols])
