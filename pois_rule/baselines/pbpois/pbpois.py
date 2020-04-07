#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 18:13:18 2018
@author: matteo
"""
"""References
    PGPE: Sehnke, Frank, et al. "Policy gradients with parameter-based exploration for
        control." International Conference on Artificial Neural Networks. Springer,
        Berlin, Heidelberg, 2008.
"""

import numpy as np
from baselines import logger
import warnings
from contextlib import contextmanager
import time
from baselines.common import colorize
import csv
import os

@contextmanager
def timed(msg, verbose=True):
    if verbose: print(colorize(msg, color='magenta'))
    tstart = time.time()
    yield
    if verbose: print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))


def eval_trajectory(env, pol, gamma, horizon, feature_fun):
    ret = disc_ret = 0
    t = 0
    ob = env.reset()
    done = False
    while not done and t < horizon:
        s = feature_fun(ob) if feature_fun else ob
        a = pol.act(s)
        ob, r, done, _ = env.step(a)
        ret += r
        disc_ret += gamma ** t * r
        t += 1

    return ret, disc_ret, t


# BINARY line search
def line_search_binary(pol, newpol, actor_params, rets, alpha, natgrad,
                       normalize=True,
                       use_rmax=True,
                       use_renyi=True,
                       max_search_ite=30, rmax=None, delta=0.2):
    rho_init = newpol.eval_params()
    low = 0.
    high = None
    bound_init, _ = newpol.eval_bound(actor_params, rets, pol, rmax,
                                   normalize, use_rmax, use_renyi, delta)
    # old_delta_bound = 0.
    rho_opt = rho_init
    i_opt = 0.
    delta_bound_opt = 0.
    epsilon_opt = 0.
    epsilon = 1.

    for i in range(max_search_ite):
        rho = rho_init + epsilon * natgrad * alpha
        newpol.set_params(rho)
        bound, _ = newpol.eval_bound(actor_params, rets, pol, rmax,
                                  normalize, use_rmax, use_renyi, delta)
        delta_bound = bound - bound_init
        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
        if np.isnan(bound) or delta_bound <= delta_bound_opt:
            high = epsilon
        else:
            low = epsilon
            rho_opt = rho
            delta_bound_opt = delta_bound
            i_opt = i
            epsilon_opt = epsilon

        old_epsilon = epsilon
        if high is None:
            epsilon *= 2
        else:
            epsilon = (low + high) / 2.
        if abs(old_epsilon - epsilon) < 1e-6:
            break

    return rho_opt, epsilon_opt, delta_bound_opt, i_opt + 1


def line_search_parabola(pol, newpol, actor_params, rets, alpha, natgrad,
                         normalize=True,
                         use_rmax=True,
                         use_renyi=True,
                         max_search_ite=30, rmax=None, delta=0.2):
    epsilon = 0.001
    epsilon_old = 0.
    max_increase = 2.
    delta_bound_tol = 1e-4
    delta_bound_old = -np.inf
    bound_init, _ = newpol.eval_bound(actor_params, rets, pol, rmax,
                                   normalize, use_rmax, use_renyi, delta)
    rho_old = rho_init = newpol.eval_params()

    for i in range(max_search_ite):

        rho = rho_init + epsilon * alpha * natgrad
        newpol.set_params(rho)

        bound, renyi = newpol.eval_bound(actor_params, rets, pol, rmax,
                                  normalize, use_rmax, use_renyi, delta)

        #print("Line search iteration %d, Bound:%f   , Renyi: %f" %(i,bound, renyi))
        if np.isnan(bound):
            warnings.warn('Got NaN bound value: rolling back!')
            return rho_old, epsilon_old, delta_bound_old, i + 1

        delta_bound = bound - bound_init

        epsilon_old = epsilon

        if delta_bound > (1. - 1. / (2 * max_increase)) * epsilon_old:
            epsilon = epsilon_old * max_increase
        else:
            epsilon = epsilon_old ** 2 / (2 * (epsilon_old - delta_bound))

        if delta_bound <= delta_bound_old + delta_bound_tol:
            if delta_bound_old < 0.:
                return rho_init, 0., 0., i + 1
            else:
                return rho_old, epsilon_old, delta_bound_old, i + 1

        delta_bound_old = delta_bound
        rho_old = rho

    return rho_old, epsilon_old, delta_bound_old, i + 1


def optimize_offline(pol, newpol, actor_params, rets, grad_tol=1e-4, bound_tol=1e-4, max_offline_ite=100,
                     normalize=True,
                     use_rmax=True,
                     use_renyi=True,
                     max_search_ite=30,
                     rmax=None, delta=0.2, use_parabola=False, verbose=True,
                     step_size=1,
                     var_step_size = 0.1):
    print('Optimizing offline')
    improvement = 0.
    rho = pol.eval_params()

    fmtstr = "%6i %10.3g %10.3g %18i %18.3g %18.3g %18.3g"
    titlestr = "%6s %10s %10s %18s %18s %18s %18s"
    if verbose: print(titlestr % ("iter", "epsilon", "step size", "num line search",
                                  "gradient norm", "delta bound ite", "delta bound tot"))

    natgrad = None

    for i in range(max_offline_ite):
        # Candidate policy
        newpol.set_params(rho)

        # Bound with gradient
        bound, grad, ret_mean, renyi, batch_size = newpol.eval_bound_and_grad(actor_params, rets, pol, rmax,
                                                 normalize, use_rmax, use_renyi, delta)

        '''print("Bound:",bound)
        print("Grad:", grad)
        print("ret_mean:", ret_mean)
        print("renyi:", renyi)
        print("batch_size:", batch_size)
        input()'''
        if np.any(np.isnan(grad)):
            warnings.warn('Got NaN gradient! Stopping!')
            return rho, improvement, i
        if np.isnan(bound):
            warnings.warn('Got NaN bound! Stopping!')
            return rho, improvement, i

            # Natural gradient
        if newpol.diagonal:
            natgrad = grad / (newpol.eval_fisher() + 1e-24)
        else:
            raise NotImplementedError
        # assert np.dot(grad, natgrad) >= 0

        grad_norm = np.sqrt(np.dot(grad, natgrad))
        print('grad:', grad)
        print('natgrad:', natgrad)
        if grad_norm < grad_tol:
            if verbose: print("stopping - gradient norm < gradient_tol - grad_norm:", grad_norm)
            if verbose > 1: print(rho)
            return rho, improvement, i

        # Step size search
        alpha = step_size / grad_norm ** 2
        alpha = np.array([1. if i < len(rho) // 2 else var_step_size for i in range(len(rho))]) * alpha  #lower learning rate for the variance
        line_search = line_search_parabola if use_parabola else line_search_binary
        rho, epsilon, delta_bound, num_line_search = line_search(pol,
                                                                 newpol,
                                                                 actor_params,
                                                                 rets,
                                                                 alpha,
                                                                 natgrad,
                                                                 normalize=normalize,
                                                                 use_rmax=use_rmax,
                                                                 use_renyi=use_renyi,
                                                                 max_search_ite=max_search_ite,
                                                                 rmax=rmax,
                                                                 delta=delta)
        newpol.set_params(rho)
        improvement += delta_bound
        if verbose: print(
                    fmtstr % (i + 1, epsilon, np.mean(alpha * epsilon), num_line_search, grad_norm, delta_bound, improvement))
        if delta_bound < bound_tol:
            if verbose: print("stopping - delta bound < bound_tol")
            if verbose > 1: print(rho)
            return rho, improvement, i

    return rho, improvement, max_offline_ite - 1


def learn(env_maker, pol_maker, eval_policy,
          gamma, num_theta, horizon, max_iters,
          feature_fun=None,
          iw_norm='sn',
          bound='max-ess',
          max_offline_iters=100,
          max_search_ite=30,
          verbose=True,
          save_weights=True,
          delta=0.2,
          rho_att=None,
          center_return=False,
          line_search_type='parabola',
          adaptive_batch=False,
          eval_episodes=24,
          eval_frequency=20,
          eval_theta_path=None,
          episodes_per_theta=1,
          sampler=None,
          save_to=None,
          step_size=1,
          var_step_size=0.1):
    # Initialization
    env = env_maker()
    pol = pol_maker('pol', env.observation_space, env.action_space)
    newpol = pol_maker('newpol', env.observation_space, env.action_space)
    newpol.set_params(pol.eval_params())
    old_rho = pol.eval_params()
    batch_size = num_theta * episodes_per_theta
    normalize = True if iw_norm == 'sn' else False
    episodes_so_far = 0
    thetas_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()
    offline_iters_counts =[]
    if bound == 'std-d2':
        use_rmax = False
        use_renyi = True
    elif bound == 'max-d2':
        use_rmax = True
        use_renyi = True
    elif bound == 'max-ess':
        use_rmax = True
        use_renyi = False
    elif bound == 'std-ess':
        use_rmax = False
        use_renyi = False
    else:
        raise NotImplementedError

    if line_search_type == 'parabola':
        use_parabola = True
    elif line_search_type == 'binary':
        use_parabola = False
    else:
        raise NotImplementedError

    promise = -np.inf
    actor_params, rets, disc_rets, lens = [], [], [], []
    old_actor_params, old_rets, old_disc_rets, old_lens = [], [], [], []
    rho_att.append('timesteps_so_far')
    best_rew = -np.inf
    if eval_theta_path is not None:

        with open(eval_theta_path, 'w', newline='') as out_file:
            dicti_writer = csv.DictWriter(out_file, fieldnames=rho_att)
            dicti_writer.writeheader()
    # Learning
    for it in range(max_iters):
        if it % eval_frequency == 0:
            print("Evaluating:")
            rho = pol.eval_params()
            rho = np.reshape(rho, (int(len(rho)/2), 2), 'F')
            pol.set_actor_params(rho[:, 0])
            def pi_wrapper(ob):
                s = feature_fun(ob) if feature_fun else ob
                return pol.act(s)

            rew, _, logs = eval_policy(pi=pi_wrapper, n_episodes=eval_episodes, verbose=True)
            # Saving best
            if rew > best_rew and save_to is not None:
                if not os.path.exists(save_to):
                    os.makedirs(save_to)
                np.save(save_to + '/best', rho)
                best_rew = rew
                logger.log('Saved policy weights as %s' % os.path.join(save_to, 'best.npy'))
            if eval_theta_path is not None:
                rho_to_save = rho
                
                with open(eval_theta_path, 'a') as out_file:
                    dicti_writer = csv.DictWriter(out_file, fieldnames=rho_att)
                    dict_row = {}
                    for row in range(len(rho_att)-1):
                        dict_row[rho_att[row]] = rho_to_save[row // 2][row % 2]
                    dict_row["timesteps_so_far"] = timesteps_so_far
                    dicti_writer.writerow(dict_row)
        logger.log('\n********** Iteration %i ************' % it)
        rho = pol.eval_params()  # Higher-order-policy parameters
        if save_to:
            if not os.path.exists(save_to):
                os.makedirs(save_to)
            np.save(save_to + '/weights', np.reshape(rho, (int(len(rho)/2), 2), 'F'))
        if verbose > 1:
            logger.log('Higher-order parameters: ', rho)
        if save_weights:
            w_to_save = rho

        # Add 100 trajectories to the batch
        with timed('Sampling', verbose):
            if sampler:
                print("Sampling Parallel")
                frozen_pol = pol.freeze()
                for ep in range(num_theta):
                    theta = frozen_pol.resample()
                    #actor_params.append(theta)
                    _rets, _disc_rets, _lens = sampler.collect(theta)
                    timesteps_so_far += sum(_lens)
                    '''lens.append(np.mean(_lens))
                    rets.append(np.mean(_rets))
                    disc_rets.append(np.mean(_disc_rets))'''
                    lens = lens + _lens
                    rets = rets + _rets
                    disc_rets = disc_rets + _disc_rets
                    actor_params = actor_params + [theta] * len(_rets)
                    timesteps_so_far += np.sum(_lens)
            else:
                frozen_pol = pol.freeze()
                for ep in range(num_theta):
                    theta = frozen_pol.resample()
                    #actor_params.append(theta)
                    ret = 0
                    disc_ret = 0
                    ep_len = 0
                    for _ in range(episodes_per_theta):
                        r, dr, el = eval_trajectory(env, frozen_pol, gamma, horizon, feature_fun)
                        '''ret += r
                        disc_ret += dr
                        ep_len += el'''
                        rets.append(r)
                        disc_rets.append(dr)
                        lens.append(el)
                        actor_params.append(theta)
                        timesteps_so_far += el

                    '''rets.append(ret / episodes_per_theta)
                    disc_rets.append(disc_ret / episodes_per_theta)
                    lens.append(ep_len / episodes_per_theta)'''

        complete = len(rets) >= batch_size  # Is the batch complete?
        # Normalize reward
        norm_disc_rets = np.array(disc_rets)
        if center_return:
            norm_disc_rets = norm_disc_rets - np.mean(norm_disc_rets)
        rmax = np.max(abs(norm_disc_rets))
        # Estimate online performance
        perf = np.mean(norm_disc_rets)
        episodes_so_far += num_theta * episodes_per_theta
        episodes_this_iter = num_theta * episodes_per_theta
        thetas_so_far += num_theta
        print("Sampled episodes!!")
        with timed('summaries before'):
            logger.log("Performance (plain, undiscounted): ", np.mean(rets[-episodes_this_iter:]))
            # Data regarding the episodes collected in this iteration
            logger.record_tabular("Iteration", it)
            logger.record_tabular("InitialBound", newpol.eval_bound(actor_params, norm_disc_rets, pol, rmax,
                                                                    normalize, use_rmax, use_renyi, delta))
            logger.record_tabular("EpLenMean", np.mean(lens[-episodes_this_iter:]))
            logger.record_tabular("EpRewMean", np.mean(norm_disc_rets[-episodes_this_iter:]))
            logger.record_tabular("UndEpRewMean", np.mean(norm_disc_rets[-episodes_this_iter:]))
            logger.record_tabular("EpThisIter", episodes_this_iter)
            logger.record_tabular("ThetasSoFar", thetas_so_far)
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("BatchSize", batch_size)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

        if adaptive_batch and complete and perf < promise and batch_size < 5 * episodes_this_iter:
            # The policy is rejected (unless batch size is already maximal)
            iter_type = 0
            if verbose: logger.log('Rejecting policy (expected at least %f, got %f instead)!\nIncreasing batch_size' %
                                   (promise, perf))
            batch_size += episodes_this_iter  # Increase batch size
            newpol.set_params(old_rho)  # Reset to last accepted policy
            promise = -np.inf  # No need to test last accepted policy
            # Reuse old trajectories
            actor_params = old_actor_params
            rets = old_rets
            disc_rets = old_disc_rets
            lens = old_lens
            if verbose: logger.log('Must collect more data (have %d/%d)' % (len(rets), batch_size))
            complete = False
        elif complete:
            # The policy is accepted, optimization is performed
            iter_type = 1
            old_rho = rho  # Save as last accepted policy (and its trajectories)
            old_actor_params = actor_params
            old_rets = rets
            old_disc_rets = disc_rets
            old_lens = lens
            with timed('offline optimization', verbose):
                rho, improvement, num_off_it = optimize_offline(pol, newpol, actor_params, norm_disc_rets,
                                                    normalize=normalize,
                                                    use_rmax=use_rmax,
                                                    use_renyi=use_renyi,
                                                    max_offline_ite=max_offline_iters,
                                                    max_search_ite=max_search_ite,
                                                    rmax=rmax,
                                                    delta=delta,
                                                    use_parabola=use_parabola,
                                                    verbose=verbose,
                                                    step_size=step_size,
                                                    var_step_size=var_step_size)
                print("Finished %d offline iterations" %num_off_it)
                offline_iters_counts.append(num_off_it)
                if save_to:
                    np.save(os.path.join(save_to, 'offline_iters'), offline_iters_counts)
                newpol.set_params(rho)
                # assert(improvement>=0.)
                # Expected performance
                promise, _ = newpol.eval_bound(actor_params, norm_disc_rets, pol, rmax,
                                            normalize, use_rmax, use_renyi, delta)
        else:
            # The batch is incomplete, more data will be collected
            iter_type = 2
            if verbose: logger.log('Must collect more data (have %d/%d)' % (len(rets), batch_size))
            newpol.set_params(rho)  # Policy stays the same

        # Save data
        if save_weights:
            logger.record_tabular('Weights', str(w_to_save))

        with timed('summaries after'):
            unn_iws = newpol.eval_iws(actor_params, behavioral=pol, normalize=False)
            iws = unn_iws / np.sum(unn_iws)
            ess = np.linalg.norm(unn_iws, 1) ** 2 / np.linalg.norm(unn_iws, 2) ** 2
            J, varJ = newpol.eval_performance(actor_params, norm_disc_rets, behavioral=pol)
            renyi = newpol.eval_renyi(pol)
            bound, _ = newpol.eval_bound(actor_params, norm_disc_rets, pol, rmax,
                                      normalize, use_rmax, use_renyi, delta)

            # Data regarding the whole batch
            logger.record_tabular('BatchSize', batch_size)
            logger.record_tabular('IterType', iter_type)
            logger.record_tabular('Bound', bound)
            # Discounted, [centered]
            logger.record_tabular('InitialReturnMean', np.mean(norm_disc_rets))
            logger.record_tabular('InitialReturnMax', np.max(norm_disc_rets))
            logger.record_tabular('InitialReturnMin', np.min(norm_disc_rets))
            logger.record_tabular('InitialReturnStd', np.std(norm_disc_rets))
            # Discounted, uncentered
            logger.record_tabular('UncReturnMean', np.mean(disc_rets))
            logger.record_tabular('UncReturnMax', np.max(disc_rets))
            logger.record_tabular('UncReturnMin', np.min(disc_rets))
            logger.record_tabular('UncReturnStd', np.std(disc_rets))
            # Undiscounted, uncentered
            logger.record_tabular('PlainReturnMean', np.mean(rets))
            logger.record_tabular('PlainReturnMax', np.max(rets))
            logger.record_tabular('PlainReturnMin', np.min(rets))
            logger.record_tabular('PlainReturnStd', np.std(rets))
            # Iws
            logger.record_tabular('D2', renyi)
            logger.record_tabular('ReturnMeanIw', J)
            logger.record_tabular('MaxIWNorm', np.max(iws))
            logger.record_tabular('MinIWNorm', np.min(iws))
            logger.record_tabular('MeanIWNorm', np.mean(iws))
            logger.record_tabular('StdIWNorm', np.std(iws))
            logger.record_tabular('MaxIW', np.max(unn_iws))
            logger.record_tabular('MinIW', np.min(unn_iws))
            logger.record_tabular('MeanIW', np.mean(unn_iws))
            logger.record_tabular('StdIW', np.std(unn_iws))
            logger.record_tabular('ESSClassic', ess)
            logger.record_tabular('ESSRenyi', batch_size / np.exp(renyi))

        logger.dump_tabular()

        # Update behavioral
        pol.set_params(newpol.eval_params())
        if complete:
            # Start new batch
            actor_params, rets, disc_rets, lens = [], [], [], []