#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:31:17 2020

@author: umberto
"""

# -*- coding: utf-8 -*-
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
import numpy as np

from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from ddpg.ddpg_driver import DDPG
from stable_baselines.common.math_util import scale_action
from gym_torcs import TorcsEnv


#%% Build staff
# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 1.,  8.,  9., 11., 14., 16., 17., 20., 45., 46., 49.,  59., 62.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)



#%% create model
n_actions = 3
param_noise = None # AdaptiveParamNoiseSpec()
#action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.05) * np.ones(n_actions))
action_noise = None#NormalActionNoise(mean=np.array([0., 0., 0.]), sigma=np.array([0.05, 0.05, 0.05]))
env = TorcsEnv(reward_function,collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=0, slow=False, faster=True, graphic=True)


model = DDPG.load("model_file/ddpg_BC_128_64")
model.env = env

#%% Let's see what happens after BC pretraining
obs = env.reset()
reward_sum = 0.0
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_sum += reward
    #env.render()
    if done:
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

env.close()