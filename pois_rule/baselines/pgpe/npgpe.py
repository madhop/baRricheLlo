#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from baselines import logger
from baselines.common.adam import Adam
"""
Created on Wed Apr  4 18:13:18 2018
@author: matteo
"""
"""References
    PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
        control." International Conference on Artificial Neural Networks. Springer,
        Berlin, Heidelberg, 2008.
    Optimal baseline: Zhao, Tingting, et al. "Analysis and
        improvement of policy gradient estimation." Advances in Neural
        Information Processing Systems. 2011.
"""


def eval_trajectory(env, pol, gamma, task_horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t < task_horizon:
        s = feature_fun(ob, env) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ret += r
        disc_ret += gamma ** t * r
        t += 1

    return ret, disc_ret, t


def learn(env, pol, gamma, step_size, batch_size, task_horizon, max_iterations,
          feature_fun=None, use_baseline=True, step_size_strategy="adam",
          verbose=True,
          save_to=None,
          natural=True,
          eval_episodes=24,
          eval_frequency=20,
          eval_theta_path=None,
          episodes_per_theta=1,
          enable_evaluation=True,
          sampler=None,
          eval_sampler=None,
          eval_policy=None,
          load=False):
    # Logging
    format_strs = []
    if verbose: format_strs.append('stdout')
    if save_to: format_strs.append('csv')
    logger.configure(dir=save_to, format_strs=format_strs)
    rho = pol.eval_params()
    adam_mean = Adam(rho.shape[0])
    adam_var = Adam(rho.shape[0])
    best_rew = -np.inf
    assert episodes_per_theta > 0
    scores = []
    if load:
        scores = np.load(save_to + '/online_scores.npy').tolist()
    # Learning iteration
    for it in range(max_iterations):
        if it % eval_frequency == 0:
            pol.set_mean()

            if enable_evaluation:
                print("Evaluation Policy at iteration: %d" % (it))
                if eval_sampler is not None:
                    rewards, _, _ = eval_sampler.collect(pol.get_theta())
                    eval_sampler.restart()
                    rew = np.mean(rewards)
                else:
                    def pi_wrapper(ob):
                        s = feature_fun(ob)
                        return pol.act(s)

                    rew, _, _ = eval_policy(pi=pi_wrapper, n_episodes=eval_episodes, verbose=True)
                # Saving best
                if rew > best_rew and save_to is not None:
                    np.save(save_to + '/best', rho)
                    best_rew = rew
                    logger.log('Saved policy weights as %s' % os.path.join(save_to, 'best.npy'))
            if eval_theta_path is not None:
                pol.save(eval_theta_path)

        rho = pol.eval_params()  # Higher-order-policy parameters
        if save_to:
            np.save(save_to + '/weights', rho)

        # Batch of episodes
        # TODO: try symmetric sampling
        print("sampling %d thetas" % batch_size)
        if sampler is None:
            actor_params = []
            rets, disc_rets, lens = [], [], []

            for ep in range(batch_size):
                theta = pol.resample()
                print("Theta:")
                print(theta)
                actor_params.append(theta)
                ret = 0
                disc_ret = 0
                ep_len = 0
                for _ in range(episodes_per_theta):
                    r, dr, el = eval_trajectory(env, pol, gamma, task_horizon, feature_fun)
                    ret += r
                    disc_ret += dr
                    ep_len += el

                rets.append(ret / episodes_per_theta)
                disc_rets.append(disc_ret / episodes_per_theta)
                lens.append(ep_len / episodes_per_theta)
        else:
            if it > 0 and it % eval_frequency == 0:
                sampler.restart()
            print("Collecting parallel")
            actor_params = []
            rets, disc_rets, lens = [], [], []
            for _ in range(batch_size):
                theta = pol.resample()
                print("Theta:")
                print(theta)
                actor_params.append(theta)
                ret, disc_ret, len = sampler.collect(theta)
                rets.append(np.mean(ret))
                disc_rets.append(np.mean(disc_ret))
                lens.append(np.mean(len))
        logger.log('\n********** Iteration %i ************' % it)
        if verbose:
            print('Higher-order parameters:', rho)
            # print('Fisher diagonal:', pol.eval_fisher())
            # print('Renyi:', pol.renyi(pol))
        logger.record_tabular('AvgRet', np.mean(rets))
        logger.record_tabular('J', np.mean(disc_rets))
        logger.record_tabular('VarJ', np.var(disc_rets, ddof=1) / batch_size)
        logger.record_tabular('BatchSize', batch_size)
        logger.record_tabular('AvgEpLen', np.mean(lens))
        scores.append((np.min(ret), np.max(ret), np.mean(ret)))
        np.save(save_to + '/online_scores.npy', scores)

        # Update higher-order policy
        if natural:
            grad = pol.eval_natural_gradient(actor_params, disc_rets, use_baseline=use_baseline)
        else:
            grad = pol.eval_gradient(actor_params, disc_rets, use_baseline=use_baseline)

        if verbose:
            print('grad:', grad)

        grad2norm = np.linalg.norm(grad, 2)
        gradmaxnorm = np.linalg.norm(grad, np.infty)

        step_size_it = {'const': step_size,
                        'norm': step_size / grad2norm if grad2norm > 0 else 0,
                        'vanish': step_size / np.sqrt(it + 1),
                        'adam': np.stack(
                            [adam_mean.update(grad[:, 0], step_size), adam_var.update(grad[:, 1], step_size)]).T
                        }.get(step_size_strategy, step_size)

        if step_size_strategy == 'adam':
            delta_rho = step_size_it
        else:
            delta_rho = step_size_it * grad

        pol.set_params(rho + delta_rho)

        logger.record_tabular('StepSize', step_size_it)
        logger.record_tabular('GradInftyNorm', gradmaxnorm)
        logger.record_tabular('Grad2Norm', grad2norm)
        logger.dump_tabular()
