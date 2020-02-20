from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
import numpy as np

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

simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')

# Find best laps from demonstrations
all_laps = np.unique(simulations.NLap)
lap_times = map(lambda lap: simulations[simulations.NLap == lap]['time'].values[-1], all_laps)
ref_time = ref_df['curLapTime'].values[-1]
perc_deltas = list(map(lambda t: (abs(t - ref_time) / ref_time * 100) <= 1.5, lap_times))
right_laps = all_laps[perc_deltas]
print('Using {} demonstrator laps'.format(len(right_laps)))

# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)

# Create dataset
expert_simulations = to_SARS(simulations[simulations.NLap.isin(right_laps)], reward_function)
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

#np.savez('trajectory/expert_demonstrations', expert_demonstrations)
#expert_ds = ExpertDataset(expert_path='trajectory/expert_demonstrations.npz')

expert_ds = ExpertDataset(traj_data=expert_demonstrations, batch_size=1000, train_fraction=0.8)

n_actions = 3
param_noise = None
action_noise = None  # NormalActionNoise(mean=np.array([0., 0., 0.]), sigma=np.array([0.05,0.05, 0.05]))
env = TorcsEnv(reward_function, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=3, slow=False, graphic=True)

policy_kwargs = {'layers': [64, 64], 'act_fun': tf.tanh}
model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, batch_size=3000,
             policy_kwargs=policy_kwargs)

model2, log = model.pretrain(expert_ds, n_epochs=50, learning_rate=1e-4)

import matplotlib.pyplot as plt
plt.plot(log['train_loss'])
plt.plot(log['val_loss'])
plt.show()
model.save('model_file/ddpg_bc')

"""obs = env.reset()
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
"""
