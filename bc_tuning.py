from joblib import Parallel, delayed
import os
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import action_cols, ref_traj_cols, penalty_cols, state_dict
import pandas as pd
import numpy as np
import argparse
import itertools
import pickle

from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.ddpg.policies import MlpPolicy
from ddpg.ddpg_driver import DDPG
from gym_torcs import TorcsEnv

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def run_experiment(state_id, policy_layers, policy_activation, batch_size, epochs, out_dir):

    state_cols = state_dict[state_id]
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

    # Create dataset
    expert_simulations = to_SARS(simulations[simulations.NLap.isin(right_laps)], reward_function, state_cols)
    expert_simulations = expert_simulations.reset_index(drop=True)

    # Create dictionary structure
    # keys: actions, obs, rewards, episode_returns, episode_starts

    starts_index = expert_simulations[expert_simulations['episode_starts']].index
    episode_returns = []
    for i in range(len(starts_index) - 1):
        episode_returns.append(expert_simulations.loc[starts_index[i]:starts_index[i + 1]]['r'].sum())
    episode_returns.append(expert_simulations.loc[starts_index[-1]:]['r'].sum())

    expert_demonstrations = dict()
    expert_demonstrations['actions'] = expert_simulations[action_cols].values
    expert_demonstrations['obs'] = expert_simulations[state_cols].values
    expert_demonstrations['rewards'] = expert_simulations['r'].values
    expert_demonstrations['episode_starts'] = expert_simulations['episode_starts']
    expert_demonstrations['episode_returns'] = np.array(episode_returns)

    expert_ds = ExpertDataset(traj_data=expert_demonstrations, batch_size=batch_size, train_fraction=0.8)

    policy_kwargs = {'layers': policy_layers, 'act_fun': policy_activation}

    env = TorcsEnv(reward_function, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
                   gear_change=False, brake=True, start_env=False, damage_th=3, slow=False, graphic=True)

    model = DDPG(MlpPolicy, env, verbose=0, param_noise=None, action_noise=None, batch_size=-5,
                 normalize_observations=True, policy_kwargs=policy_kwargs)

    model, log = model.pretrain(expert_ds, n_epochs=epochs, val_interval=100)

    activation_name = 'tanh' if policy_activation == tf.nn.tanh else 'relu'
    name = 'ddpgbc_{}_{}_{}_{}_{}'.format(state_id, policy_layers, activation_name, batch_size, epochs)
    model.save(out_dir + name)
    pickle.dump(log, open(out_dir + 'log_' + name + '.pkl', 'wb'))
    print('Saved {}'.format(name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', nargs='+', type=int)
    parser.add_argument('--layers', nargs='+', type=int, action='append')
    parser.add_argument('--activation', nargs='+')
    parser.add_argument('--epochs', nargs='+', type=int)
    #parser.add_argument('--learning_rate', nargs='+', type=np.float64)
    parser.add_argument('--state_type', nargs='+', type=int, default=0)
    parser.add_argument('--out_dir', type=str, default='../ddpg_bc/')
    parser.add_argument('--n_jobs', type=int, default=-1)
    args = parser.parse_args()
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    activations = [tf.nn.tanh if a == 'tanh' else tf.nn.relu for a in args.activation]

    params = [args.state_type, args.layers, activations, args.batch_size, args.epochs]#, args.learning_rate]
    params_comb = list(itertools.product(*params))

    print('Started experiments')
    #Parallel(prefer="threads", n_jobs=args.n_jobs)(delayed(run_experiment)(*x, out_dir) for x in params_comb)
    Parallel(n_jobs=args.n_jobs)(delayed(run_experiment)(*x, out_dir) for x in params_comb)
    print('Experiments terminated')
