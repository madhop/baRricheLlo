import gym
from gym import spaces
import snakeoil3_gym as snakeoil3
import numpy as np
import pandas as pd
import copy
import collections as col
import os
import time
from utils_torcs import *


class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 300

    initial_reset = True


    def __init__(self, reward_function, vision=False, throttle=False, gear_change=False, brake=False):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.brake = brake
        self.reward_function = reward_function

        self.initial_run = True

        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            os.system('torcs  -nofuel -nodamage -nolaptime &')
            #os.system('torcs -r &')
        time.sleep(0.5)
        os.system('sh autostart.sh')    # autostart Practice
        time.sleep(0.5)

        """
        obs = client.S.d  # Get the current full-observation from torcs
        """

    def step(self, u):
        #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d
        #print('action_torcs', action_torcs)

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
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
            action_torcs['gear'] = self.auto_gear_snakeoil(client.S.d['speed_x'])

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

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # Reward setting Here TODO #######################################
        # direction-dependent positive reward
        #track = np.array(obs['track'])
        sp = np.array(obs['speed_x'])
        progress = sp*np.cos(obs['angle'])
        reward = progress
        """sp = np.array(obs['speed_x'])
        progress = sp*np.cos(obs['angle'])
        d = {'xCarWorld': self.observation['x'], 'yCarWorld': self.observation['y']}
        reward = self.reward_function(pd.DataFrame(d, index = [0]))
        print('reward:', reward)"""

        # Termination judgement #########################
        episode_terminate = False

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 4:
            print('Hit wall')
            episode_terminate = True
            reward = 0

        """if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                #client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            #client.R.d['meta'] = True

        """


        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        #return self.get_obs(), reward, client.R.d['meta'], {}
        return self.get_obs(), reward, episode_terminate, {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

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

    def make_observaton(self, raw_obs):
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
