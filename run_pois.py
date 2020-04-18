#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:29:07 2020

@author: umberto
"""
import os
import sys

import argparse
import pandas as pd
import numpy as np
from pois_rule.policies.weight_hyperpolicy import PeRuleBasedPolicy
from pois_rule.baselines.pbpois import pbpois as pois
from pois_rule.policies.eval_policy import eval_policy
import pois_rule.baselines.common.tf_util as U
from pois_rule.baselines import logger
import pickle
from fqi.reward_function import *
from fqi.utils import *
from controller import MeanController
from gym_torcs_ctrl import TorcsEnv
import time
init_logstd = 0.1
"""default_values = np.array(
        [[np.log(0.5), init_logstd],
         [np.log(0.02), init_logstd],
         [np.log(5), init_logstd],
         [np.log(0.055), init_logstd],
         [np.log(3.), init_logstd],
         [np.log(73.5), init_logstd],
         [np.log(116), init_logstd]]
    )"""
"""
-0.69076616,	-3.88496,	1.6196413,	-2.8960278,	1.1091019,	4.314237,	4.7429957	       -210.762518135787
-0.65601856,	-3.9396932,	1.610632,	-2.9221127,	1.0915695,	4.269636,	4.743861	       -179.220295066277
-0.7106042,     -3.789705,  1.57127,	-2.9225132,	1.1347715,	4.3603606,	4.6552763          -206.887423424249


"""

default_means = np.array([0.5, 0.02, 5, 0.055, 3., 73.5, 116])
default_means = np.log(default_means)
#default_logstd = np.ones(7)*init_logstd
#default_logstd = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
default_means = np.array([-0.7106042,     -3.789705,  1.57127,	-2.9225132,	1.1347715,	4.3603606,	4.6552763])
default_logstd = np.array([0.01, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

def starter(x):
    return [0, 0, 1, 7]

def run_experiment(num_iterations, timestep_length, horizon, delta=0.2, out_dir='.',
                   parallel=False, episodes_per_scenario=1, verbose=True, num_theta=1, num_workers=5, punish_jerk=True,
                   gamma=0.999, eval_frequency=10, eval_episodes=20, scale_reward=1., jerk_pun=0.5, hsd_pun=2.,
                   continuous=False, po=False, three_actions=False, collision_penalty=1000, **alg_args):
    
    
    ref_df = pd.read_csv('trajectory/ref_traj.csv')
    reward_function = Spatial_projection(ref_df, penalty=None)
    #reward_function = Temporal_projection(ref_df, penalty=None)

    env = TorcsEnv(reward_function, collision_penalty=-collision_penalty, low_speed_penalty=-10000, state_cols=state_cols, ref_df=ref_df, vision=False,
                   throttle=True, gear_change=False, brake=True, start_env=False, damage_th=10.0, slow=False,
                   faster=False, graphic=False, starter=starter)
    
    C = MeanController(ref_df)
    
    policyConstructor = PeRuleBasedPolicy
    n_actions = 3
    action_closure = C.action_closure
    
    def make_policy(name, b=0, c=0):
        pi = policyConstructor(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                                 means_init=default_means, #default_values[:, 0],
                                 logstds_init=default_logstd, #default_values[:, 1],
                                 verbose=verbose,
                                 n_actions=n_actions,
                                 action_closure=action_closure)
        return pi
    
    sampler = None
    sess = U.single_threaded_session()
    sess.__enter__()
    
    def eval_policy_closure(**args):
        return eval_policy(env=env, **args)
    
    def make_env():
        return env
    
    rho_att = ['alpha1_mean','alpha1_var', 'alpha2_mean','alpha2_var', 'speed_y_thr_mean','speed_y_thr_var', 
               'beta1_mean','beta1_var', 'gamma1_mean','gamma1_var', 'gamma2_mean','gamma2_var',
               'gamma3_mean', 'gamma3_var']
    
    time_str = str(int(time.time())) + '_' + str(num_theta) + '_' + str(delta) + '_' + str(collision_penalty)
    #logger.configure(dir=out_dir + '/logs', format_strs=['stdout', 'csv', 'tensorboard', 'json'], suffix=time_str)
    pois.learn(make_env, make_policy, num_theta=num_theta, horizon=horizon,
               max_iters=num_iterations, delta=delta, sampler=sampler, feature_fun=None,
               line_search_type='parabola', gamma=gamma, eval_frequency=eval_frequency,
               eval_episodes=eval_episodes, eval_policy=eval_policy_closure,
               episodes_per_theta=episodes_per_scenario,
               rho_att = rho_att,
               eval_theta_path=out_dir + '/logs' + '/eval_theta_episodes-' + time_str + '.csv',
               save_to=out_dir + '/models/'+time_str+'/',
               out_dir=out_dir,
               logger_suffix=time_str,
               **alg_args)

#%% run experiment
run_experiment(num_iterations=10000, 
               timestep_length=0.1,
               horizon=1000,
               eval_episodes=1,
               out_dir='POIS_logs',
               verbose=True,
               num_theta=200,
               delta=0.2, #0.001, #
               eval_frequency=1,
               collision_penalty=2000)
