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


#%% Build staff and env
# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 9., 14., 16., 17., 20., 47., 49., 55., 59., 60., 61., 62., 63., 65., 68.])
#right_laps = np.array([ 1.,  8.,  9., 11., 14., 16., 17., 20., 45., 46., 49.,  59., 62.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)
env = TorcsEnv(reward_function,collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=0, slow=False, faster=False, graphic=True)


#%% create model
n_actions = 3
param_noise = None # AdaptiveParamNoiseSpec()
"""action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0., -1.1, 0.8]), dt=1,
                                            sigma=np.array([0.05, 0.1, 0.1]), theta=0.1, initial_noise=[0., -2., 1])"""

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0., 0., 0.]),
                                            theta=.02, 
                                            dt=1e-2,
                                            sigma=np.array([0.02, 0.05, 0.05]),
                                            initial_noise=[0., -3., 2.])

model = DDPG(MlpPolicy, env, gamma=0.9999, verbose=1, nb_rollout_steps=5, nb_train_steps=1, normalize_observations=True,
             param_noise=param_noise, action_noise=action_noise, 
             batch_size=32, policy_kwargs={'layers': [150, 300]})

#%% expert BC
# Create dataset
#expert_simulations = to_SARS(simulations[simulations.NLap.isin(right_laps)], reward_function)
expert_simulations = to_SARS(simulations, reward_function)
expert_simulations = expert_simulations.reset_index(drop=True)

# Create dictionary structure
# keys: actions, obs, rewards, episode_returns, episode_starts

starts_index = expert_simulations[expert_simulations['episode_starts']].index
episode_returns = []
for i in range(len(starts_index)-1):
    episode_returns.append(expert_simulations.loc[starts_index[i]:starts_index[i+1]]['r'].sum())
episode_returns.append(expert_simulations.loc[starts_index[-1]:]['r'].sum())

expert_demonstrations = dict()
expert_demonstrations['actions'] = expert_simulations[action_cols].values
expert_demonstrations['obs'] = expert_simulations[state_cols].values
expert_demonstrations['rewards'] = expert_simulations['r'].values
expert_demonstrations['episode_starts'] = expert_simulations['episode_starts']
expert_demonstrations['episode_returns'] = np.array(episode_returns)
expert_ds = ExpertDataset(traj_data=expert_demonstrations, batch_size=2000)

model.pretrain(expert_ds, n_epochs=2000)

#%% Save model after BC
model.save('model_file/ddpg_BC_128_64')

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

#%% DDPG online
model.learn(total_timesteps=400000, output_name='9999_150_300')
model.save('model_file/ddpg_online')

#%% BC and online

model.pretrain(expert_ds, n_epochs=500)
model.save('model_file/ddpg_BC')
model.learn(total_timesteps=400000)
model.save('model_file/ddpg_online')

#%% load model and train online
#'../bc_training_200302/start_demonstrations/ddpgbc_0_[64, 64]_tanh_3500_20000_1_1_1.zip'
model = DDPG.load("../bc_ddpg/start_demonstrations/ddpgbc_0_[64, 64]_tanh_3500_20000_1_1_1.zip")
model.nb_rollout_steps = 5
model.verbose = 1
model.env = env
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0., 0., 0.]),
                                            theta=.02, 
                                            dt=1e-2,
                                            sigma=np.array([0.02, 0.05, 0.05]),
                                            initial_noise=[0., -3., 2.])
model.action_noise = action_noise
model.batch_size = 32

model.learn(total_timesteps=400000)
model.save('model_file/ddpg_online')