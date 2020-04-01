import os
import sys
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
import random
from gym.spaces import Box, Discrete, Tuple
from baselines import logger
from env.cross_sumo_env import CrossSumoEnv
from env.cross_sumo_env_continuous import CrossSumoEnvContinuous
from policies.eval_policy import eval_policy
from policies.rule_based_policy import extract_features
from policies.cross.rule_based_5w import act_closure as act_closure_6_levels, RuleBasedPolicyCross6Levels
from policies.cross.rule_based_cross_2_levels import act_closure as act_closure_2_levels, RuleBasedPolicyCross2Levels
from policies.cross.weight_hyperpolicy import PeRuleBasedPolicy
from policies.cross.weight_hyperpolicy import PeRuleBasedPolicy as PeRuleBasedPolicyContinuous
import time
import argparse
import numpy as np
# Framework imports
import gym
import tensorflow as tf
# Self imports: utils
from baselines.common import set_global_seeds
import baselines.common.tf_util as U
# Self imports: algorithm
from baselines.pbpois import pbpois as pois
from baselines.pbpois.parallel_sampler import ParallelSampler

default_values = np.array(
        [[300., 3.],
         [-20, 1.],
         [4, 1.5],
         [4, 1.5],
         [4, 1.5],
         [4, 1.5],
         [4, 1.5],
         [7, 1.5],
         [7, 1.5]]
    )

# noinspection PyInterpreter,PyInterpreter
def run_experiment(num_iterations, timestep_length, perception_delay, action_delay, port, seed, horizon, out_dir='.',
                   parallel=False, episodes_per_scenario=1, verbose=True, num_theta=10, num_workers=5, punish_jerk=True,
                   gamma=0.999, eval_frequency=10, eval_episodes=20, scale_reward=1., jerk_pun=0.5, hsd_pun=2.,
                   continuous=False, po=False, three_actions=False, **alg_args):

    start_time = time.time()
    step_delay_p = int(round(perception_delay / timestep_length))
    step_delay_a = int(round(action_delay / timestep_length))
    time_str = str(start_time)
    directory_output = 'cross/pois_PO_' + str(step_delay_p) + '_' + str(step_delay_a)
    if continuous:
        directory_output += '/continuous'
    else:
        directory_output += '/discrete'
    directory_output += '/models/' + time_str+'/'

    env_params = dict(
        mode='cl',
        horizon=horizon,
        scale_state=True,
        time_step=timestep_length,
        delay_perc=perception_delay,
        delay_ac=action_delay,
        scale_reward=scale_reward,
        start_time=start_time,
        directory_output=directory_output,
        punish_jerk=punish_jerk,
        jerk_pun=jerk_pun,
        hsd_pun=hsd_pun,
        po=po)

    n_actions=3
    action_closure = None
    if continuous:
        raise ValueError("Continuous Policy Not Implemented")
        envConstructor = CrossSumoEnvContinuous
        policyConstructor = PeRuleBasedPolicyContinuous
    elif three_actions:
        envConstructor = CrossSumoEnv
        policyConstructor = PeRuleBasedPolicy
        n_actions = 3
        action_closure = act_closure_2_levels
        default_values = RuleBasedPolicyCross2Levels.default_values
        rho_att = RuleBasedPolicyCross2Levels.rho_att
    else:
        envConstructor = CrossSumoEnv
        policyConstructor = PeRuleBasedPolicy
        n_actions = 8
        action_closure = act_closure_6_levels
        default_values = RuleBasedPolicyCross6Levels.default_values
        rho_att = RuleBasedPolicyCross6Levels.rho_att

    env = envConstructor(flag_eval=False, port=port, seed=seed, **env_params)
    env_eval = envConstructor(port=port + 1, seed=seed + 1684, flag_eval=True, **env_params)
    port += 4

    def make_env(i=35):
        return envConstructor(port=port + 4 * i, seed=seed + 100 * i, flag_eval=False, **env_params)

    def eval_policy_closure(**args):
        return eval_policy(env=env_eval, **args)

    def feature_fun_closure(ob):
        return extract_features(ob, env=env)

    ob_space = Box(low=-np.inf, high=np.inf, shape=(12,))
    ac_space = Discrete(8)
    num_cfg_files = 5



    def make_policy(name, b=0, c=0):
        pi = policyConstructor(name=name, ob_space=ob_space, ac_space=ac_space,
                                 means_init=default_values[:, 0],
                                 logstds_init=default_values[:, 1],
                                 verbose=verbose,
                                 n_actions=n_actions,
                                 action_closure=action_closure)
        return pi

    sampler = None
    if parallel:
        sampler = ParallelSampler(make_policy, make_env, n_workers=num_workers, horizon=horizon,
                               gamma=gamma, feature_fun=feature_fun_closure,
                               seed=seed, episodes_per_worker=episodes_per_scenario)
        sess = U.make_session(num_cpu=num_cfg_files)
        sess.__enter__()
    else:
        sess = U.single_threaded_session()
        sess.__enter__()
    np.random.seed(seed)
    random.seed(seed)
    time_str = str(start_time)
    logger.configure(dir=out_dir + '/' + directory_output + '/logs',
                         format_strs=['stdout', 'csv'], suffix=time_str)

    pois.learn(make_env, make_policy, num_theta=num_theta, horizon=horizon,
               max_iters=num_iterations, sampler=sampler, feature_fun=feature_fun_closure,
               line_search_type='parabola', gamma=gamma, eval_frequency=eval_frequency,
               eval_episodes=eval_episodes, eval_policy=eval_policy_closure,
               episodes_per_theta=episodes_per_scenario,
               rho_att = rho_att,
               eval_theta_path=directory_output + '/logs' + '/eval_theta_episodes-' + time_str + '.csv',
               save_to=out_dir + '/' + directory_output + '/models/'+time_str+'/', **alg_args)
    print('TOTAL TIME:', time.time() - start_time)
    print("Time taken: %f seg" % ((time.time() - start_time)))
    print("Time taken: %f hours" % ((time.time() - start_time)/3600))

    env.close()


if __name__ == '__main__':
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
