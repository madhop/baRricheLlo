from gym_torcs import TorcsEnv
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
import numpy as np
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from ddpg.ddpg_driver import DDPG
from stable_baselines.common.math_util import scale_action

from stable_baselines.deepq.replay_buffer import PrioritizedReplayBuffer

ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
simulations = simulations[simulations.NLap == 17]
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 1.,  8.,  9., 11., 14., 16., 17., 20., 45., 46., 49.,  59., 62.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)
dataset = to_SARS(simulations, reward_function)

env = TorcsEnv(reward_function, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=3, slow=False, graphic=True)

PRB = PrioritizedReplayBuffer(10, 0.005)
batch_samples = list(zip(dataset[state_cols].values,
                         dataset[action_cols].values,
                         dataset['r'].values,
                         dataset[state_prime_cols].values,
                         dataset['absorbing'].values))

print('Fill Replay Buffer')
for t in batch_samples:
    scaled_a = scale_action(env.action_space, t[1])
    PRB.add(t[0], scaled_a, *t[2:])
