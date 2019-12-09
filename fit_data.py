import pandas as pd
import sys
import os
import pickle
import argparse
from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from sklearn.ensemble.forest import ExtraTreesRegressor
from trlib.algorithms.reinforcement.fqi_driver import FQIDriver, DoubleFQIDriver
from trlib.environments.trackEnv import TrackEnv
from trlib.utilities.ActionDispatcher import *
from fqi.dataset_preprocessing import *
"""from fqi.fqi_evaluate import run_evaluation
from fqi.et_tuning import run_tuning"""
from fqi.utils import *
from fqi.reward_function import *
from fqi.sars_creator import *
sys.setrecursionlimit(3000)

ref_df = pd.read_csv('./trajectory/ref_traj.csv')
data_df = pd.read_csv('./trajectory/dataset.csv')

def run_experiment(track_file_name, rt_file_name, data_path, max_iterations, output_path, n_throttle,
               n_brake, n_steer, n_jobs, output_name, reward_function, delta_t,
               filter_actions, ad_type, tuning, kdt_norm, kdt_param, filt_a_outliers, double_fqi, evaluation):


    # build SARS
    reward_function = Speed_projection(ref_df)
    sars_data = to_SARS(data_df, reward_function)
    """dataset = prepare_dataset(os.path.join('./trajectory/dataset.csv'),
                              os.path.join('./trajectory/ref_traj.csv'),
                              reward_function=reward_function, delta_t=delta_t)"""

    print('SARS prepared')
    nmin = 5

    # Create environment
    state_dim = len(state_cols)
    action_dim = len(action_cols)
    mdp = TrackEnv(state_dim, action_dim, 0.99999, 'continuous')

    # Parameters of ET regressor
    regressor_params = {'n_estimators': 100,
                        'criterion': 'mse',
                        'min_samples_split': 2,
                        'min_samples_leaf': nmin,
                        'n_jobs': n_jobs,
                        'random_state': 42}
    regressor = ExtraTreesRegressor

    if ad_type == 'fkdt':
        action_dispatcher = FixedKDTActionDispatcher
        alg_actions = sars_data[action_cols].values

    elif ad_type == 'rkdt':
        action_dispatcher = RadialKDTActionDispatcher
        alg_actions = sars_data[action_cols].values

    elif ad_type == 'discrete':
        action_dispatcher = ConstantActionDispatcher
        actions, sub_actions = create_action_combinations(sars_data, n_throttle, n_brake, n_steer, filter_actions)
        alg_actions = sub_actions
    else:
        action_dispatcher = None
        alg_actions = None

    # Create policy instance
    epsilon = 0  # no exploration
    pi = EpsilonGreedy([], ZeroQ(), epsilon)

    # Define the order of the columns to pass to the algorithm
    # state_prime_cols: colonne dello stato successivo
    cols = ['t'] + state_cols + action_cols + ['r'] + state_prime_cols + ['absorbing']
    # Define the masks used by the action dispatcher
    state_mask = [i for i, s in enumerate(state_cols) if s in knn_state_cols]
    data_mask = [i for i, c in enumerate(cols) if c in knn_state_cols]

    if double_fqi:
        fqi = DoubleFQIDriver
    else:
        fqi = FQIDriver

    algorithm = fqi(mdp=mdp,
                    policy=pi,
                    actions=alg_actions,
                    max_iterations=max_iterations,
                    regressor_type=regressor,
                    data=sars_data[cols].values,
                    action_dispatcher=action_dispatcher,
                    state_mask=state_mask,
                    data_mask=data_mask,
                    s_norm=kdt_norm,
                    filter_a_outliers=filt_a_outliers,
                    ad_n_jobs=n_jobs,
                    ad_param=kdt_param,
                    verbose=True,
                    **regressor_params)

    result = algorithm.step()

    # save algorithm object
    algorithm_name = output_name + '.pkl'
    with open(output_path + '/' + algorithm_name, 'wb') as output:
        pickle.dump(algorithm, output, pickle.HIGHEST_PROTOCOL)

    # save action dispatcher object
    AD_name = 'AD_' + algorithm_name
    with open(output_path + '/' + AD_name, 'wb') as output:
        pickle.dump(algorithm._action_dispatcher, output, pickle.HIGHEST_PROTOCOL)
    print('Saved Action Dispatcher')


run_experiment('dataset', 'ref_traj', './trajectory/', 100, './model_file/', 3,3,3, 10, 'first_model', 'progress', 2, False, 'rkdt', False, False, 10, False, True, False)
