import argparse
import pickle
from sklearn.ensemble.forest import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from utils import state_cols, action_cols
import numpy as np
import math
import os


file_path = os.path.dirname(os.path.abspath(__file__))


def run_tuning(dataset, nmin, half, n_jobs=1, output_path='', output_name='', track_file_name='', rt_file_name='',
               data_path=''):

    if len(dataset) == 0:
        # Create dataset
        dataset, _ = prepare_dataset(os.path.join(data_path, track_file_name + '.csv'),
                                     os.path.join(data_path, rt_file_name + '.csv'),
                                     reward_function='progress', knn_actions=True)

    X = dataset[state_cols + action_cols].values
    t = dataset['r'].values
    n_samples = len(t)
    ids = list(range(n_samples))

    if half:
        np.random.shuffle(ids)
        ids_A = ids[:math.floor(n_samples/2)]
        ids_B = ids[math.floor(n_samples/2):]
    else:
        ids_A = ids

    mdl = ExtraTreesRegressor(n_estimators=100, criterion='mse', n_jobs=n_jobs)

    gcv = GridSearchCV(mdl, {'min_samples_leaf': nmin}, cv=10, scoring='neg_mean_squared_error')
    # Fit the models
    gcv.fit(X[ids_A, :], t[ids_A])

    if half:
        gcv_list = []
        gcv_list.append(gcv)

        gcv = GridSearchCV(mdl, {'min_samples_leaf': nmin}, cv=10, scoring='neg_mean_squared_error')
        # Fit the models
        gcv.fit(X[ids_B], t[ids_B])
        gcv_list.append(gcv)
        to_save = gcv_list
    else:
        to_save = gcv

    if output_path != '':

        # Save the results
        with open(os.path.join(output_path, output_name+'.pkl'), 'wb') as out:
            pickle.dump(to_save, out, pickle.HIGHEST_PROTOCOL)
        print('Saved cross val results as {}'.format(output_name))

    if half:
        return gcv_list
    else:
        return gcv


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--track_file_name", type=str, help='Name of the data file containing the laps')
    parser.add_argument("--rt_file_name", type=str, help='Name of the file containing reference trajectory')
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(file_path, '..', '..', '..', '..', '..', '..', '..', 'data',
                                             'ferrari', 'driver', 'datasets', 'csv'),
                        help='Path of the folder containing csv data files')
    parser.add_argument("--nmin", nargs='+', type=int)
    parser.add_argument("--output_path", type=str, default=os.path.join('..', 'fqi_experiments'),
                        help='Path to save results')
    parser.add_argument("--output_name", type=str, default='et_tune')
    parser.add_argument("--n_jobs", type=int, default=1)

    args = parser.parse_args()
    out_dir = args.output_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_tuning([], args.nmin, args.n_jobs, args.output_path, args.output_name, args.track_file_name, args.rt_file_name,
               args.data_path)
