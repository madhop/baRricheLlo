from joblib import Parallel, delayed
import os
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import action_cols, ref_traj_cols, penalty_cols, state_dict, bound_dict
import pandas as pd
import numpy as np
import argparse
import itertools
from fqi.direction_feature import  create_direction_feature

from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines import DDPG
from n_gym_torcs import NTorcsEnv

import tensorflow as tf


def run_experiment(state_id, policy_layers, policy_activation, batch_size, epochs, action_weights, out_dir):

    state_cols = state_dict[state_id]
    low = bound_dict[state_id]['low']
    high = bound_dict[state_id]['high']

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

    # Reward function
    penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
    penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
    reward_function = Temporal_projection(ref_df, penalty=penalty)

    state_bounds = {'low': low, 'high': high}
    env = NTorcsEnv(reward_function, state_cols=state_cols, state_bound=state_bounds, ref_df=ref_df, vision=False,
                    throttle=True,
                    gear_change=False, brake=True, start_env=False, damage_th=3, slow=False, graphic=True)

    # Create dataset
    simulations = create_direction_feature(simulations)
    expert_simulations = to_SARS(simulations[simulations.NLap.isin(right_laps)], reward_function, state_cols)
    expert_simulations = expert_simulations.reset_index(drop=True)

    # Create dictionary structure
    # keys: actions, obs, rewards, episode_returns, episode_starts

    starts_index = expert_simulations[expert_simulations['episode_starts']].index
    episode_returns = []
    for i in range(len(starts_index) - 1):
        episode_returns.append(expert_simulations.loc[starts_index[i]:starts_index[i + 1]]['r'].sum())
    episode_returns.append(expert_simulations.loc[starts_index[-1]:]['r'].sum())

    # Scale state and actions to [0,1] and [-1,1] respectively for the DDPG algorithm
    expert_demonstrations = dict()
    expert_demonstrations['actions'] = env.scale_action(expert_simulations[action_cols].values)
    expert_demonstrations['obs'] = env.scale_obs(expert_simulations[state_cols].values)
    expert_demonstrations['rewards'] = expert_simulations['r'].values
    expert_demonstrations['episode_starts'] = expert_simulations['episode_starts']
    expert_demonstrations['episode_returns'] = np.array(episode_returns)

    expert_ds = ExpertDataset(traj_data=expert_demonstrations, batch_size=batch_size, train_fraction=0.8)

    policy_kwargs = {'layers': policy_layers, 'act_fun': policy_activation}

    model = DDPG(MlpPolicy, env, verbose=1, param_noise=None, action_noise=None, batch_size=2000,
                 normalize_observations=False, policy_kwargs=policy_kwargs, seed=1)

    #model, log = model.pretrain(expert_ds, n_epochs=epochs, val_interval=100, action_weights=action_weights)
    model = model.pretrain(expert_ds, n_epochs=epochs, val_interval=100)

    activation_name = 'tanh' if policy_activation == tf.nn.tanh else 'relu'
    if action_weights:
        aw = '{}_{}_{}'.format(*action_weights)
    else:
        aw = '1_1_1'
    name = 'ddpgbc_{}_{}_{}_{}_{}_{}'.format(state_id, policy_layers, activation_name, batch_size, epochs, aw)
    model.save(out_dir + name)
    #pickle.dump(log, open(out_dir + 'log_' + name + '.pkl', 'wb'))
    print('Saved {}'.format(name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', nargs='+', type=int)
    parser.add_argument('--layers', nargs='+', type=int, action='append')
    parser.add_argument('--activation', nargs='+')
    parser.add_argument('--epochs', nargs='+', type=int)
    parser.add_argument('--state_type', nargs='+', type=int, default=0)
    parser.add_argument('--action_weights', nargs='+', type=np.float32, action='append', default=[None])
    parser.add_argument('--out_dir', type=str, default='../ddpg_bc/')
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    activations = [tf.nn.tanh if a == 'tanh' else tf.nn.relu for a in args.activation]

    params = [args.state_type, args.layers, activations, args.batch_size, args.epochs, args.action_weights]
    params_comb = list(itertools.product(*params))

    print('Started experiments')
    Parallel(prefer="threads", n_jobs=args.n_jobs)(delayed(run_experiment)(*x, out_dir) for x in params_comb)
    #Parallel(n_jobs=args.n_jobs)(delayed(run_experiment)(*x, out_dir) for x in params_comb)
    print('Experiments terminated')
