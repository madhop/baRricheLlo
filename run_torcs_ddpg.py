from gym_torcs_std import TorcsEnv
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from ddpg.ddpg_driver import DDPG
from stable_baselines.common.math_util import scale_action

# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 1.,  8.,  9., 11., 14., 16., 17., 20., 45., 46., 49.,  59., 62.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)

dataset = to_SARS(simulations, reward_function)

env = TorcsEnv(reward_function,collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=0, slow=False, graphic=True)

"""model = DDPG.load("model_file/ddpg_99")
model.env = env
batch_samples = list(zip(dataset[state_cols].values,
                         dataset[action_cols].values,
                         dataset['r'].values,
                         dataset[state_prime_cols].values,
                         dataset['absorbing'].values))
for t in batch_samples:
    scaled_a = scale_action(model.action_space, t[1])
    model.replay_buffer.add(t[0], scaled_a, *t[2:])

model.nb_rollout_steps = 400"""

n_actions = 3
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))
#action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=np.array([0.5, 0.1, 0.1]))

#model = DDPG(MlpPolicy, env, nb_rollout_steps=3000, verbose=1, param_noise=param_noise, action_noise=action_noise, buffer_size=50000, batch_size=128)
#model = DDPG(MlpPolicy, env, nb_train_steps=1, nb_rollout_steps=3000, verbose=1, param_noise=param_noise, action_noise=action_noise, buffer_size=50000, batch_size=512)
"""print('Adding demonstrations to replay buffer')
batch_samples = list(zip(dataset[state_cols].values,
                         dataset[action_cols].values,
                         dataset['r'].values,
                         dataset[state_prime_cols].values,
                         dataset['absorbing'].values))
for t in batch_samples:
    scaled_a = scale_action(model.action_space, t[1])
    model.replay_buffer.add(t[0], scaled_a, *t[2:])"""

#model.learn(log_interval=5000, total_timesteps=60000, episode_count=1, save_buffer=False, save_model=True)
#model.save('model_file/ddpg_online')

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, render=True)
model.learn(total_timesteps=400000)
