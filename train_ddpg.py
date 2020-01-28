from gym_torcs import TorcsEnv

import time
import os
import sys
import tensorflow as tf
import numpy as np


from utils_torcs import *
from preprocess_raw_torcs_algo import *
from build_dataset_offroad import *
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from ddpg.ddpg_driver import DDPG

# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

reward_function = Temporal_projection(ref_df)
simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 1.,  8.,  9., 11., 14., 16., 17., 20., 45., 46., 49.,  59., 62.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
rf = Temporal_projection(ref_df, penalty=penalty)

dataset = to_SARS(simulations, rf)

env = TorcsEnv(reward_function, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False)
n_actions = 3
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)

batch_samples = list(zip(dataset[state_cols].values,
                         dataset[action_cols].values,
                         dataset['r'].values,
                         dataset[state_prime_cols].values,
                         dataset['absorbing'].values))

print('Started batch pretraining')
model.batch_pretraining(batch_samples, max_iterations=10, tol=1e-20)
print('Finished batch pretraining')
model.save("model_file/ddpg_torcs")
print('Model saved.')

# model.learn(total_timesteps=2000)
