from gym_torcs_std import TorcsEnv
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from ddpg.ddpg_driver import DDPG
from stable_baselines.common.math_util import scale_action

#%% Build model
n_actions = 3
param_noise = AdaptiveParamNoiseSpec()  # None
env = TorcsEnv(vision=False, throttle=True, gear_change=False)

action_noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0., 0., 0.]),
                                            theta=.02,  
                                            dt=1e-2,
                                            sigma=np.array([0.02, 0.05, 0.05]),
                                            initial_noise=[0., 2., -3.])

model = DDPG(MlpPolicy, env, gamma=0.9999, verbose=1, nb_rollout_steps=5, nb_train_steps=1, normalize_observations=True,
             param_noise=param_noise, action_noise=action_noise,
             batch_size=32, policy_kwargs={'layers': [64, 64]})

#% learn
model.learn(total_timesteps=400000)
