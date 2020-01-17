import sys
import os
import pickle
import argparse
import pandas as pd
from sklearn.ensemble.forest import ExtraTreesRegressor

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '..'))
sys.setrecursionlimit(3000)

from fqi.et_tuning import run_tuning
from data_processing.sars.reward_function import *
from data_processing.sars.sars_creator import to_SARS
from fqi.utils import *

from trlib.policies.valuebased import EpsilonGreedy
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi_driver import FQIDriver, DoubleFQIDriver
from trlib.environments.trackEnv import TrackEnv
from trlib.utilities.ActionDispatcher import *

from fqi.fqi_evaluate import run_evaluation

def run_experiment(track_file_name, rt_file_name, data_path, max_iterations, output_path, n_jobs,
                   output_name, reward_function, r_penalty, rp_kernel, rp_band, ad_type, tuning,
                   tuning_file_name, kdt_norm, kdt_param, filt_a_outliers, double_fqi, evaluation, first_step):

    # Load dataset and refernce trajectory
    print('Loading data')
    simulations = pd.read_csv(os.path.join(data_path, track_file_name + '.csv'),
                              dtype={'isReference': bool, 'is_partial':bool})
    ref_tr = pd.read_csv(os.path.join(data_path, rt_file_name + '.csv'))

    if r_penalty:
        print('Computing penalty')

        # Take as training laps the set of laps with lap time lower than the 1.5% of the reference trajectory
        # lap time
        all_laps = np.unique(simulations.NLap)
        lap_times = map(lambda lap: simulations[simulations.NLap == lap]['time'].values[-1], all_laps)
        ref_time = ref_tr['time'].values[-1]
        perc_deltas = list(map(lambda t: (abs(t - ref_time) / ref_time * 100) <= 1.5, lap_times))
        right_laps = all_laps[perc_deltas]

        p_params = {}
        if rp_band is not None:
            p_params['bandwidth'] = rp_band
        if rp_kernel is not None:
            p_params['kernel'] = rp_kernel

        penalty = LikelihoodPenalty(**p_params)
        penalty.fit(simulations[simulations.NLap.isin(right_laps)][state_cols].values)

        if reward_function == 'temporal':
            rf = Temporal_projection(ref_tr, penalty=penalty, clip_range=(-np.inf, np.inf))
        elif reward_function == 'discrete':
            rf = Discrete_temporal_reward(ref_tr, penalty=penalty, clip_range=(-np.inf, np.inf))
        elif reward_function == 'distance':
            rf = Spatial_projection(ref_tr, penalty=penalty, clip_range=(-np.inf, np.inf))
        elif reward_function == 'speed':
            rf = Speed_projection(ref_tr, penalty=penalty, clip_range=(-np.inf, np.inf))
        elif reward_function == 'curv':
            rf = Curv_temporal(ref_tr, penalty=penalty, clip_range=(-np.inf, np.inf))
    else:
        if reward_function == 'temporal':
            rf = Temporal_projection(ref_tr)
        elif reward_function == 'discrete':
            rf = Discrete_temporal_reward(ref_tr)
        elif reward_function == 'distance':
            rf = Spatial_projection(ref_tr)
        elif reward_function == 'speed':
            rf = Speed_projection(ref_tr)
        elif reward_function == 'curv':
            rf = Curv_temporal(ref_tr)

    dataset = to_SARS(simulations, rf)

    nmin = 1

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


    if first_step:
        # Create new policy instance
        if policy_type == 'greedy':
            epsilon = 0
            pi = EpsilonGreedy([], ZeroQ(), epsilon)
        elif policy_type == 'boltzmann':
            temperature = 0.5  # no exploration
            pi = Softmax([], ZeroQ(), temperature)
    else:
        # import policy
        algorithm_name = output_name + '.pkl'
        policy_name = 'policy_' + algorithm_name
        with open(output_path + '/' + policy_name, 'rb') as pol:
            print('loading policy')
            pi = pickle.load(pol)
            print('pi:', pi)

    # Define the order of the columns to pass to the algorithm
    cols = ['t'] + state_cols + action_cols + ['r'] + state_prime_cols + ['absorbing']
    # Define the masks used by the action dispatcher
    state_mask = [i for i, s in enumerate(state_cols) if s in knn_state_cols]
    data_mask = [i for i, c in enumerate(cols) if c in knn_state_cols]

    if ad_type == 'fkdt':
        action_dispatcher = FixedKDTActionDispatcher
        alg_actions = dataset[action_cols].values

    elif ad_type == 'rkdt':
        action_dispatcher = RadialKDTActionDispatcher
        alg_actions = dataset[action_cols].values

    else:
        action_dispatcher = None
        alg_actions = None

    if double_fqi:
        fqi = DoubleFQIDriver
    else:
        fqi = FQIDriver

    algorithm = fqi(mdp=mdp,
                    policy=pi,
                    actions=alg_actions,
                    max_iterations=max_iterations,
                    regressor_type=regressor,
                    data=dataset[cols].values,
                    action_dispatcher=action_dispatcher,
                    state_mask=state_mask,
                    data_mask=data_mask,
                    s_norm=kdt_norm,
                    filter_a_outliers=filt_a_outliers,
                    ad_n_jobs=n_jobs,
                    ad_param=kdt_param,
                    verbose=True,
                    **regressor_params)

    print('Starting execution')
    algorithm.step()

    # save algorithm object
    algorithm_name = output_name + '.pkl'
    with open(output_path + '/' + algorithm_name, 'wb') as output:
        pickle.dump(algorithm, output, pickle.HIGHEST_PROTOCOL)

    # save policy object
    policy_name = 'policy_' + algorithm_name
    with open(output_path + '/' + policy_name, 'wb') as output:
        pickle.dump(algorithm._policy, output, pickle.HIGHEST_PROTOCOL)
    print('Saved policy object')

    # save action dispatcher object
    AD_name = 'AD_' + algorithm_name
    with open(output_path + '/' + AD_name, 'wb') as output:
        pickle.dump(algorithm._action_dispatcher, output, pickle.HIGHEST_PROTOCOL)
    print('Saved Action Dispatcher')

    if evaluation:

        print('Evaluation')
        run_evaluation(output_path+'/'+algorithm_name, track_file_name, data_path, n_jobs, output_path,
                       'eval_'+output_name, False,
                       output_path + '/' + AD_name)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--track_file_name", type=str,
                        help='Name of the data file containing the simulation laps')

    parser.add_argument("--rt_file_name", type=str,
                        help='Name of the file containing reference trajectory 100Hz')

    parser.add_argument("--data_path", type=str,
                        default=os.path.join(file_path, '..', '..', '..', '..', '..', '..', '..', 'data',
                                             'ferrari', 'driver', 'datasets', 'same_setup', 'csv'),
                        help='Path of the folder containing csv data files')

    parser.add_argument("--max_iterations", type=int, default=100, help='Number of iterations')

    parser.add_argument("--output_path", type=str, default=os.path.join('..', 'fqi_experiments'),
                        help='Path to save results')

    parser.add_argument("--output_name", type=str, default='algorithm', help='Name of the output file')

    parser.add_argument("--n_jobs", type=int, default=1)

    parser.add_argument("--reward", type=str, default='temporal',
                        help='Type of the reward [temporal, discrete, speed, distance, curv]')

    parser.add_argument("--r_penalty", action='store_true', help='To add penalty term to the reward')

    parser.add_argument("--rp_kernel", type=str, default=None, help='Kernel used for KDE')

    parser.add_argument("--rp_band", type=float, default=None, help='Bandwidth used for KDE')

    parser.add_argument("--ad_type", type=str, default='',
                        help='Type of action dispatcher, i.e., of KNN performed.' +
                        ' [fkdt, rkdt, discrete] for fixed K, Radial or fixed action set respectively')

    parser.add_argument("--tuning", action='store_true', help='To perform tuning of the Extratree model')

    parser.add_argument("--tuning_file_name", type=str, default='', help='Tuning file name')

    parser.add_argument("--kdt_norm", action='store_true',
                        help='Normalize the states feature for the neighbor search')

    parser.add_argument("--kdt_param", type=float, default=100,
                        help='Parameter of the KDT, k for fkdt and r for rkdt')

    parser.add_argument("--a_filt_out", action='store_true',
                        help='To filter the outliers in the action set of each state.')

    parser.add_argument("--double_fqi", action='store_true',
                        help='To perform double FQI instead of the standard version')

    parser.add_argument("--no_evaluate", action='store_false', help='To not perform the evaluation')

    args = parser.parse_args()
    out_dir = args.output_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_experiment(args.track_file_name, args.rt_file_name, args.data_path, args.max_iterations,
                   args.output_path, args.n_jobs, args.output_name, args.reward, args.r_penalty,
                   args.rp_kernel, args.rp_band, args.ad_type, args.tuning, args.tuning_file_name,
                   args.kdt_norm, args.kdt_param, args.a_filt_out, args.double_fqi, args.no_evaluate)
