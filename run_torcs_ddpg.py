from gym_torcs_std import TorcsEnv
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from ddpg.ddpg_driver import DDPG
from stable_baselines.common.math_util import scale_action


n_actions = 3
param_noise = AdaptiveParamNoiseSpec()  # None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))
env = TorcsEnv(vision=False, throttle=True, gear_change=False)

model = DDPG(MlpPolicy, env, verbose=1, nb_rollout_steps=100,  param_noise=param_noise, action_noise=action_noise, batch_size=128,
             policy_kwargs={'layers': [300, 600]})

model.learn(total_timesteps=400000)

#%% 
print(env.observation_space)