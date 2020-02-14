"""import gym
from gym_torcs import TorcsEnv
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.math_util import scale_action

env = gym.make('MountainCarContinuous-v0')

n_actions = 1
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, nb_train_steps=100, nb_rollout_steps=1000, verbose=1, param_noise=param_noise, action_noise=action_noise, buffer_size=50000, batch_size=512, render=True)
model.learn(log_interval=5000, total_timesteps=60000, episode_count=3, save_buffer=False, save_model=True)"""


import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
#from stable_baselines import DDPG
from ddpg.ddpg_driver import DDPG

env = gym.make('MountainCarContinuous-v0')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=True, seed=816)
model.learn(total_timesteps=400000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
