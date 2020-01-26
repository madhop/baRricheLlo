import sys
import os
import pickle
import argparse
import pandas as pd
from sklearn.ensemble.forest import ExtraTreesRegressor

from fqi.et_tuning import run_tuning
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *

from trlib.policies.valuebased import EpsilonGreedy, Softmax
from trlib.policies.qfunction import ZeroQ
from trlib.algorithms.reinforcement.fqi_driver import FQIDriver, DoubleFQIDriver
from trlib.environments.trackEnv import TrackEnv
from trlib.utilities.ActionDispatcher import *

from fqi.fqi_evaluate import run_evaluation
from fqi.fqi_computation import run_experiment


track_file_name = 'dataset_offroad'
rt_file_name = 'ref_traj'
data_path = './trajectory/'
max_iterations = 100
output_path = './model_file/'
n_jobs = 10

r_penalty = True
r_offroad_penalty = True

reward_function = 'temporal'
output_name = reward_function + ('_penalty_xy' if r_penalty else '') + '_reward_model'#'first_model'


rp_kernel = 'gaussian'#'exponential'
rp_band = 1#1.4384#0.88586679

filter_actions = False
filt_a_outliers = False
policy_type = 'greedy'#'boltzmann'
evaluation = False

first_step = False
output_name
print(output_name)

run_experiment(track_file_name, rt_file_name, data_path, max_iterations, output_path, n_jobs,
               output_name, reward_function, r_penalty, r_offroad_penalty, rp_kernel, rp_band, 'rkdt', False,
               '', False, 10, filt_a_outliers, True, policy_type, evaluation, first_step)
