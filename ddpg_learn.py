from torcs_environment import TORCS
from fqi.reward_function import Temporal_projection, LikelihoodPenalty
from fqi.utils import action_cols, prev_action_cols, penalty_cols
import pandas as pd
import numpy as np
from behavioural_cloning.create_expert_dataset import create_expert_dataset
import os
from stable_baselines.ddpg.policies import MlpPolicy
from ddpg.ddpg_driver import DDPG
from stable_baselines.common.noise import NormalActionNoise
import tensorflow as tf
import pickle

out_dir = '../learning_200312/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

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
         'delta_direction_x': {'low': -1, 'high': 1}, 'delta_direction_y': {'low': -1, 'high': 1}}

state_cols = list(state.keys())
state_space = {'high': np.array([state[k]['high'] for k in state_cols]),
               'low': np.array([state[k]['low'] for k in state_cols])}

print('state_cols:', state_cols)

# --- Reference trajectory and expert demonstrations
# load reference trajectory
ref_df = pd.read_csv('../demonstrations/extracted_features/ref_traj.csv')

demos_path = '../demonstrations/extracted_features/top_demonstrations.csv'
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
penalty.fit(pd.read_csv(demos_path)[penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)

batch_size = 3500
train_fraction = 0.8
demonstrations = create_expert_dataset(demos_path, reward_function, state_cols, action_cols, batch_size,
                                       train_fraction)

# --- Environment
practice_path = os.path.expanduser('~/.torcs/config/raceman/practice.xml')
env = TORCS(reward_function, state_cols, state_space, ref_df, practice_path, gear_change=False, graphic=False,
            verbose=False)

# --- RL algorithm
action_noise = NormalActionNoise(mean=np.array([0, 0, 0]), sigma=np.array([0.01, 0.05, 0.05]))
policy_kwargs = {'layers': [64, 64], 'act_fun': tf.nn.tanh}

model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, normalize_observations=True,
             policy_kwargs=policy_kwargs, gamma=0.9999, nb_train_steps=10, nb_rollout_steps=100,
             batch_size=200, n_cpu_tf_sess=None, seed=42, buffer_size=2000)

# --- Pre-training with behavioural cloning
model, log = model.pretrain(demonstrations, n_epochs=15000, val_interval=50, early_stopping=False, patience=2)
model.save(os.path.join(out_dir, 'model_bc.zip'))
pickle.dump(log, open(os.path.join(out_dir, 'log_bc_training.pkl'), 'wb'))

# --- Online learning
print('Online learning')
model.learn(10000, log_interval=50, output_name='model_int', log_path=out_dir)
model.save(os.path.join(out_dir, 'model_final.zip'))
print('Computation terminated')
