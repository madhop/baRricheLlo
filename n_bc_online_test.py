from n_gym_torcs import NTorcsEnv
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
import pandas as pd
from stable_baselines.ddpg.policies import MlpPolicy
#from ddpg.ddpg_driver import DDPG
from stable_baselines import DDPG
from fqi.utils import action_cols, ref_traj_cols, penalty_cols, state_dict, bound_dict

import tensorflow as tf

state_id = 0
state_cols = state_dict[state_id]
low = bound_dict[state_id]['low']
high = bound_dict[state_id]['high']

# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
all_laps = np.unique(simulations.NLap)
lap_times = map(lambda lap: simulations[simulations.NLap == lap]['time'].values[-1], all_laps)
ref_time = ref_df['curLapTime'].values[-1]
perc_deltas = list(map(lambda t: (abs(t - ref_time) / ref_time * 100) <= 1.5, lap_times))
right_laps = all_laps[perc_deltas]
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)

dataset = to_SARS(simulations, reward_function)
state_bounds = {'low': low, 'high': high}

env = NTorcsEnv(reward_function, state_cols=state_cols, state_bound=state_bounds, ref_df=ref_df, vision=False,
                throttle=True,
                gear_change=False, brake=True, start_env=False, damage_th=3, slow=False, graphic=True)

model = DDPG.load("../sb_pretraining/ddpgbc_0_[64, 64]_tanh_3000_15000_1_1_1.zip")

#policy_kwargs = {'layers': [64, 64], 'act_fun': tf.nn.tanh}
#model = DDPG(MlpPolicy, env, verbose=1, param_noise=None, action_noise=None, batch_size=2000,
#             normalize_observations=False, policy_kwargs=policy_kwargs, seed=1)
model.env = env


obs = env.reset()
reward_sum = 0.0
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_sum += reward
    #env.render()
    if done:
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

env.close()
