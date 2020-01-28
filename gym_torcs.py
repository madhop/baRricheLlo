import gym
from gym import spaces
import snakeoil3_gym as snakeoil3
import numpy as np
import pandas as pd
import copy
import compute_state_features as csf
from scipy import spatial
import os
import time
from utils_torcs import *


class TorcsEnv(gym.Env):
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 300

    initial_reset = True

    def __init__(self, reward_function, state_cols, ref_df, vision=False, throttle=False, gear_change=False,
                 brake=False, start_env=True, track_length=5783.85, damage_th=4):
        # print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.brake = brake
        self.reward_function = reward_function

        self.track_length = track_length
        self.damage_th = damage_th

        # save variables used for feature extraction
        self.state_cols = state_cols
        self.ref_df = ref_df
        self.tree = spatial.KDTree(list(zip(ref_df['xCarWorld'], ref_df['yCarWorld'])))

        # Create action space
        if gear_change:
            high = np.array([1., 1., 1., 7])
            low = np.array([-1., 0., 0., 1])
        else:
            high = np.array([1., 1., 1.])
            low = np.array([-1., 0., 0.])

        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Create observation space
        state_dim = len(state_cols)

        high = np.ones(state_dim) * np.inf
        low = np.ones(state_dim) * (-np.inf)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Create observation variables that comes from make_observations method
        # they are dictionaries containing raw torcs measurements
        self.observation = None
        self.p1_observation = None
        self.p2_observation = None

        # Create state variable
        self.state = None

        self.initial_run = True
        if start_env:
            self.start_env()
        """
        obs = client.S.d  # Get the current full-observation from torcs
        """

    def start_env(self):
        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            os.system('torcs  -nofuel -nodamage -nolaptime &')
            # os.system('torcs -r &')
        time.sleep(0.5)
        os.system('sh autostart.sh')  # autostart Practice
        time.sleep(0.5)

    def step(self, u, raw=False):
        #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d
        #print('action_torcs', action_torcs)

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        # Simple Automatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speed_x'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speed_x'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speed_x']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            # self.auto_gear_snakeoil(client.S.d['speed_x'])
            # return 0 to call automatic gear change
            action_torcs['gear'] = 0

        # In the case of autodriver to exit from the pitstop we set the gear to 7
        if raw:
            action_torcs['gear'] = 7

        # braking
        if self.brake is True:
            action_torcs['brake'] = this_action['brake']

        #print(action_torcs)
        # Save the privious full-obs from torcs to check if hit wall
        obs_pre = copy.deepcopy(client.S.d)

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
            # Compute the next state
            next_state = self.observation_to_state(self.make_observation(obs), self.observation, self.p1_observation, u)

            data = pd.DataFrame(data=[current_state + [0], next_state + [obs['trackPos']]], columns=self.state_cols + ['trackPos'])
            reward = self.reward_function(data)

        # Save u as previous action for the next step
        self.prev_u = u

        # Save the old observations
        self.p2_observation = self.p1_observation
        self.p1_observation = self.observation
        # Make an observation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)

        # Termination judgement #########################
        if (obs['distFromStart'] < 50) and (not raw):
            # we passed the start line
            episode_terminate = True
        else:
            episode_terminate = False

        # collision detection
        if obs['damage'] - obs_pre['damage'] > self.damage_th:
            print('Hit wall')
            collision = True
            reward = 0
        else:
            collision = False

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        # return self.get_obs(), reward, client.R.d['meta'], {}

        # The episode ends when the car reaches the start or when a collision has occurred
        # however, the episode is considered a "success" only if the start is reached
        info = {'collision': collision, 'is_success': episode_terminate}
        done = episode_terminate or collision

        if raw:
            # If raw then return current observation otherwise return the state
            return self.get_obs(), reward, done, info

        else:
            # Transform raw torcs data to state
            self.state = self.observation_to_state(self.observation, self.p1_observation, self.p2_observation, u)
            return self.get_state(), reward, done, info

    def reset(self, relaunch=False):
        # print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            # TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
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

        noise1 = 0  # (np.random.rand()-0.5)*0.002
        noise2 = 0  # (np.random.rand()-0.5)*0.002
        auto_drive = True
        start_line = False

        os.system('sh slow_time_down.sh')
        time.sleep(0.5)
        print('Started auto driving')
        while auto_drive:
            ob_distFromStart = self.observation['distFromStart']

            if ob_distFromStart < 100 and not start_line:  # just passed start line
                print('Start Line')
                start_line = True
                action = [0, 0, 1, 7]
                # Save old observations and get new one
                self.observation, _, done, _ = self.step(action, raw=True)
            elif ob_distFromStart < 5650.26 and not start_line:  # exit from pit stop
                # print('-', j)
                action = [0.012 + noise1, 0, 1, 7]
                # Save old observations and get new one
                self.observation, _, done, _ = self.step(action, raw=True)
            elif ob_distFromStart < 5703.24 and not start_line:
                # print('--', j)
                action = [-0.033 + noise2, 0, 1, 7]
                # Save old observations and get new one
                self.observation, _, done, _ = self.step(action, raw=True)
            elif ob_distFromStart < self.track_length and not start_line:
                # print('---', j)
                action = [0, 0, 1, 7]
                # Save old observations and get new one
                self.observation, _, done, _ = self.step(action, raw=True)
            else:
                # Start with agent driver and return the current state
                auto_drive = False
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
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nodamage -nolaptime &')
            #os.system('torcs -T -nofuel -nodamage -nolaptime &')
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
        ap = np.array([p['Acceleration_x'], p['Acceleration_y']])
        ar = np.array([r['Acceleration_x'], r['Acceleration_y']])
        a_actual_module, a_ref_module, a_diff_module, a_diff_of_modules, a_angle = csf.velocity_acceleration(ap, ar)
        rel_p, rho, theta = csf.position(np.array([r['xCarWorld'], r['yCarWorld']]),
                                         np.array([r1['xCarWorld'], r1['yCarWorld']]), np.array([p['x'], p['y']]))
        actual_c = csf.curvature(np.array([p['x'], p['y']]), np.array([p_1['x'], p_1['y']]),
                                 np.array([p_2['x'], p_2['y']]))
        ref_c = csf.curvature(np.array([r1['xCarWorld'], r1['yCarWorld']]), np.array([r['xCarWorld'], r['yCarWorld']]),
                              np.array([r_1['xCarWorld'], r_1['yCarWorld']]))

        state_features = {'xCarWorld': p['x'], 'yCarWorld': p['y'], 'nYawBody': p['yaw'], 'nEngine': p['rpm'],
                          'NGear': p['Gear'],
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
                          'prevaSteerWheel': prev_action[0], 'prevpBrakeF': prev_action[2],
                          'prevrThrottlePedal': prev_action[1]}

        observation = pd.DataFrame()
        for k in self.state_cols:
            observation.loc[0, k] = state_features[k]

        return observation.values

    def auto_gear_snakeoil(self, speed_x):
        #  Automatic Gear Change by Snakeoil is possible
        gear = 1
        if speed_x > 50:
            gear = 2
        if speed_x > 80:
            gear = 3
        if speed_x > 110:
            gear = 4
        if speed_x > 140:
            gear = 5
        if speed_x > 170:
            gear = 6
        if speed_x > 250:
            gear = 7
        return gear
