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
from torcs_environment import TORCS
import os
from data_processing.torcs_preprocessing import torcs_observation_to_state
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

reward_function = Temporal_projection(ref_df)

n_actions = 3
param_noise = None
action_noise = None#NormalActionNoise(mean=np.array([0., 0., 0.]), sigma=np.array([0.05,0.05, 0.05]))


# --- State definition
state = {'xCarWorld': {'low': 0, 'high': 2500}, 'yCarWorld': {'low': 0, 'high': 1200},
         'nYawBody': {'low': -np.pi, 'high': np.pi}, 'nEngine': {'low': 0, 'high': 21000},
         'positionRho': {'low': 0, 'high': 50}, 'positionTheta': {'low': -np.pi, 'high': np.pi},
         'speed_x': {'low': 0, 'high': 340}, 'speed_y': {'low': -90, 'high': 160},
         'acceleration_x': {'low': -50, 'high': 50}, 'acceleration_y': {'low': -75, 'high': 85},
         'direction_x': {'low': -1, 'high': 1}, 'direction_y': {'low': -1, 'high': 1},
         'NGear': {'low': 0, 'high': 7}, 'prevaSteerWheel': {'low': -1, 'high': 1},
         'prevpBrakeF': {'low': 0, 'high': 1}, 'prevrThrottlePedal': {'low': 0, 'high': 1},
         'delta_speed_x': {'low': -340, 'high': 340}, 'delta_speed_y': {'low': -250, 'high': 250},
         'delta_acc_x': {'low': -100, 'high': 100}, 'delta_acc_y': {'low': -160, 'high': 160},
         'delta_direction_x': {'low': -1, 'high': 1}, 'delta_direction_y': {'low': -1, 'high': 1},
         'trackPos': {'low': -1.5, 'high': 1.5}}

state_cols = list(state.keys())
state_space = {'high': np.array([state[k]['high'] for k in state_cols]),
               'low': np.array([state[k]['low'] for k in state_cols])}

practice_path = os.path.expanduser('~/.torcs/config/raceman/practice.xml')
env = TORCS(reward_function, state_cols, state_space, ref_df, practice_path, gear_change=False, graphic=True,
            verbose=True, obs_to_state_func=torcs_observation_to_state)
# model = DDPG(MlpPolicy, env, verbose=0, param_noise=None, action_noise=None, normalize_observations=True)
model = DDPG.load('../learning_200312/model_bc.zip', env=env)
#model.env = env
episode = {'obs': list(), 'reward': list(), 'done': list()}
reward_sum = 0.0
done = False
obs = env.reset()

while not done:

    action, _ = model.predict(obs)

    obs, reward, done, _ = env.step(action)
    episode['obs'].append(obs)
    episode['reward'].append(reward)
    episode['done'].append(done)
    reward_sum += reward
#pickle.dump(episode, open('../sync_tests/modifica_sleep.pkl', 'wb'))
print('Terminated')
# env.close()
