from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
import numpy as np
import pickle
import time

from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise
from ddpg.ddpg_driver import DDPG
from gym_torcs import TorcsEnv

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

reward_function = Temporal_projection(ref_df)

n_actions = 3
param_noise = None
action_noise = None#NormalActionNoise(mean=np.array([0., 0., 0.]), sigma=np.array([0.05,0.05, 0.05]))

env = TorcsEnv(reward_function, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=3, slow=False, graphic=True)

# model = DDPG(MlpPolicy, env, verbose=0, param_noise=None, action_noise=None, normalize_observations=True)
model = DDPG.load('../training_200302/start_demonstrations/ddpgbc_0_[64, 64]_tanh_3500_20000_1_1_1.zip')
episode = {'obs': list(), 'reward': list(), 'done': list()}
reward_sum = 0.0
done = False
obs = env.reset()
i = 0
while not done:
    if i == 0:
        time.sleep(0.1)
        i = 1
    #action, _ = model.predict(obs)
    if obs[1] >= 1172:
        action = [-0.02, 0, 1]
    elif obs[1] <= 1168:
        action = [0.02, 0, 1]
    else:
        action = [0, 0, 1]
    obs, reward, done, _ = env.step(action)
    episode['obs'].append(obs)
    episode['reward'].append(reward)
    episode['done'].append(done)
    reward_sum += reward
    #env.render()
    #if done:
    #print(reward_sum)
    #reward_sum = 0.0
    #obs = env.reset()
pickle.dump(episode, open('../sync_tests/modifica_sleep.pkl', 'wb'))
print('Saved')
# env.close()
