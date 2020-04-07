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
import pickle
from fqi.reward_function import *
from fqi.utils import *
import pandas as pd
from controller import MeanController
from gym_torcs_ctrl import TorcsEnv
import time

default_values = np.array(
        [[0.5, 1],
         [0.02, 1.],
         [5, 1],
         [0.055, 1],
         [3., 1],
         [73.5, 1],
         [116, 1]]
    )

def starter(x):
    return [0, 0, 1, 7]

def run_experiment(num_iterations, timestep_length, horizon, out_dir='.',
                   parallel=False, episodes_per_scenario=1, verbose=True, num_theta=1, num_workers=5, punish_jerk=True,
                   gamma=0.999, eval_frequency=10, eval_episodes=20, scale_reward=1., jerk_pun=0.5, hsd_pun=2.,
                   continuous=False, po=False, three_actions=False, **alg_args):
    
    
    ref_df = pd.read_csv('trajectory/ref_traj.csv')
    reward_function = Spatial_projection(ref_df, penalty=None)
    #reward_function = Temporal_projection(ref_df, penalty=None)

    env = TorcsEnv(reward_function, collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False,
                   throttle=True, gear_change=False, brake=True, start_env=False, damage_th=10.0, slow=False,
                   faster=False, graphic=True, starter=starter)
    
    C = MeanController(ref_df)
    
    policyConstructor = PeRuleBasedPolicy
    n_actions = 3
    action_closure = C.action_closure
    
    def make_policy(name, b=0, c=0):
        pi = policyConstructor(name=name, ob_space=env.observation_space, ac_space=env.action_space,
                                 means_init=default_values[:, 0],
                                 logstds_init=default_values[:, 1],
                                 verbose=verbose,
                                 n_actions=n_actions,
                                 action_closure=action_closure)
        return pi
    
    sampler = None
    """sess = U.single_threaded_session()
    sess.__enter__()"""
    
    def eval_policy_closure(**args):
        return eval_policy(env=env, **args)
    
    def make_env():
        return env
    
    rho_att = []
    
    time_str = str(int(time.time()))
    pois.learn(make_env, make_policy, num_theta=num_theta, horizon=horizon,
               max_iters=num_iterations, sampler=sampler, feature_fun=None,
               line_search_type='parabola', gamma=gamma, eval_frequency=eval_frequency,
               eval_episodes=eval_episodes, eval_policy=eval_policy_closure,
               episodes_per_theta=episodes_per_scenario,
               rho_att = rho_att,
               eval_theta_path=out_dir + '/logs' + '/eval_theta_episodes-' + time_str + '.csv',
               save_to=out_dir + '/models/'+time_str+'/', 
               **alg_args)

#%% run experiment
run_experiment(num_iterations=10000, 
               timestep_length=0.1,
               horizon=600,
               eval_episodes=1,
               out_dir='POIS_logs', 
               verbose=2)



#%% main
"""if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iterations", type=int, default=10000,
                         help='Maximum number of timesteps')
    parser.add_argument("--timestep_length", type=float, default=0.1,
                        help='time elapsed between two steps (def 0.1)')
    parser.add_argument("--perception_delay", type=float, default=0.0,
                        help='how much time after the actual perception the system can use it (def 0.0)')
    parser.add_argument("--action_delay", type=float, default=0.0,
                        help='how much time after the actual decision the action is performed (def 0.0)')
    parser.add_argument("--port", type=int, default=54325, help='TCP port')
    parser.add_argument("--seed", type=int, default=8, help='Random seed')
    parser.add_argument("--jerk_pun", type=float, default=0.25, help='punishment for jerk')
    parser.add_argument("--hsd_pun", type=float, default=2., help='punishment for harsh slow down')
    parser.add_argument("--scale_reward", type=float, default=1., help='Factor to scale reward function')
    parser.add_argument('--horizon', type=int, help='horizon length for episode', default=600)
    parser.add_argument('--episodes_per_scenario', type=int, help='Train episodes per scenario in a batch', default=4)
    parser.add_argument('--eval_frequency', type=int, help='Number of iterations to perform policy evaluation', default=20)
    parser.add_argument('--eval_episodes', type=int, help='Number of evaluation episodes', default=100)
    parser.add_argument('--num_theta', type=int, help='Batch size of gradient step', default=10)
    parser.add_argument('--num_workers', type=int, help='Number of parallel samplers', default=5)
    parser.add_argument('--dir', help='directory where to save data', default='.')
    parser.add_argument('--lr_strategy', help='', default='const', choices=['const', 'adam'])
    parser.add_argument('--parallel', action='store_true', help='Whether to run parallel sampler')
    parser.add_argument('--verbose', action='store_true', help='Print log messages')
    parser.add_argument('--punish_jerk', action='store_true', help='Punish jerk in the reward function')
    parser.add_argument('--continuous', action='store_true', help='Use continuous deceleration')
    parser.add_argument('--po', action='store_true', help='partial observability')
    parser.add_argument('--three_actions', action='store_true', help='partial observability')
    parser.add_argument('--iw_norm', type=str, default='sn')
    parser.add_argument('--bound', type=str, default='max-d2')
    parser.add_argument('--adaptive_batch', type=int, default=0)
    parser.add_argument('--delta', type=float, default=0.8)
    parser.add_argument('--step_size', type=float, default=1)
    parser.add_argument('--var_step_size', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--max_offline_iters', type=int, default=10)
    args = parser.parse_args()
    out_dir = args.dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    run_experiment(
                   num_iterations=args.max_iterations,
                   timestep_length=args.timestep_length,
                   perception_delay=args.perception_delay,
                   action_delay=args.action_delay,
                   port=args.port,
                   seed=args.seed,
                   horizon=args.horizon,
                   out_dir=out_dir,
                   parallel=args.parallel,
                   episodes_per_scenario=args.episodes_per_scenario,
                   verbose=args.verbose,
                   num_theta=args.num_theta,
                   num_workers=args.num_workers,
                   punish_jerk=args.punish_jerk,
                   three_actions=args.three_actions,
                   scale_reward=args.scale_reward,
                   jerk_pun=args.jerk_pun,
                   hsd_pun=args.hsd_pun,
                   continuous=args.continuous,
                   po=args.po,
                   eval_frequency=args.eval_frequency,
                   eval_episodes=args.eval_episodes,
                   iw_norm=args.iw_norm,
                   bound=args.bound,
                   delta=args.delta,
                   gamma=args.gamma,
                   max_offline_iters=args.max_offline_iters,
                   adaptive_batch=args.adaptive_batch,
                   step_size=args.step_size,
                   var_step_size=args.var_step_size

                   )
"""