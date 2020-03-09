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

out_dir = '../learning_200309/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# --- State definition
state_cols = ['xCarWorld', 'yCarWorld', 'nYawBody', 'nEngine', 'positionRho', 'positionTheta',
              'referenceCurvature', 'actualCurvature', 'actualSpeedModule', 'speedDifferenceVectorModule',
              'actualAccelerationX', 'actualAccelerationY', 'accelerationDiffX',
              'accelerationDiffY', 'direction_x', 'direction_y'] + prev_action_cols
high = np.array(
    [2500., 15000., np.pi, 21000., 2500., np.pi, np.pi, np.pi, 340., 340., 25., 85., 50., 70., 1., 1.,
     1., 10., 10.])
low = np.array(
    [0., 0., -np.pi, 0., 0., -np.pi, -np.pi, -np.pi, 0., 0., -55., -75., -60., -90., -1., 0., 0., -10, -10])

state_space = {'high': high,
               'low': low}

print('state_cols:', state_cols)

# --- Reference trajectory and expert demonstrations
# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')

demos_path = 'trajectory/demonstrations.csv'
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
penalty.fit(pd.read_csv(demos_path)[penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)

batch_size = 3500
train_fraction = 0.8
demonstrations = create_expert_dataset(demos_path, reward_function, state_cols, action_cols, batch_size,
                                       train_fraction)

# --- Environment
practice_path = os.path.expanduser('~/.torcs/config/raceman/practice.xml')
env = TORCS(reward_function, state_cols, state_space, ref_df, practice_path, gear_change=False, graphic=False)

# --- RL algorithm
action_noise = NormalActionNoise(mean=np.array([0, 0, 0]), sigma=np.array([0.01, 0.05, 0.05]))
policy_kwargs = {'layers': [64, 64], 'act_fun': tf.nn.tanh}

model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise, normalize_observations=True,
             policy_kwargs=policy_kwargs, gamma=0.9999, nb_train_steps=100, nb_rollout_steps=200,
             batch_size=1000, n_cpu_tf_sess=None, seed=42, buffer_size=50000)

# --- Pre-training with behavioural cloning
model, log = model.pretrain(demonstrations, n_epochs=30000, val_interval=50, early_stopping=False, patience=2)
model.save(os.path.join(out_dir, 'model_bc.zip'))
pickle.dump(log, open(os.path.join(out_dir, 'log_bc_training.pkl'), 'wb'))

# --- Online learning
print('Online learning')
model.learn(5000, log_interval=200, output_name='model_int', log_path=out_dir)
model.save(os.path.join(out_dir, 'model_final.zip'))
print('Computation terminated')
