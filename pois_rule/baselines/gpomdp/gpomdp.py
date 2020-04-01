import numpy as np
import os
from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.adam import Adam
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy
from tqdm import tqdm
import csv

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    pbar = tqdm(total=horizon)

    while True:
        ac, vpred, _, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews,  "new" : news,
                    "ac" : acs,
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred, _, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            pbar.n = 0
            pbar.refresh()
            cur_ep_ret = 0
            cur_ep_len = 0
            env.reset()

        i = t % horizon
        obs[i] = ob
        news[i] = new
        acs[i] = ac
        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1
        pbar.update(i - pbar.n)


def compute_gradient(pi, x_dataset,y_dataset, r_dataset, dones_dataset,
                       episode_length, discount_f=0.9999, features_idx=[0],
                       omega=None, normalize_f=False, verbose=False,
                       use_baseline=False, use_mask=False, scale_features=1, behavioral_pi=None, seed=None):
    steps = len(x_dataset)
    r_dim = len(features_idx)
    logger = {}
    # discount factor vector over a finite episode
    gamma = []
    for t in range(episode_length):
        gamma.append(discount_f ** t)
    if seed is not None:
        np.random.seed(seed)
    # discounted reward features computation
    if dones_dataset:
        num_episodes = np.sum(dones_dataset)
    else:
        num_episodes = int(steps // episode_length)
    if verbose:
        print("Episodes:", num_episodes)

    discounted_phi = []
    for episode in range(num_episodes):
        base = episode * episode_length
        r = np.array(r_dataset[base: base + episode_length]).T
        # b = np.random.binomial(1, 1, size=EPISODE_LENGTH)
        reward = []
        try:
            for idx in features_idx:
                reward.append(r[idx])
        except:
            print("Episode:", episode)
            raise ValueError("Dataset corrupted")
        reward = np.array(reward).T * scale_features
        if omega is not None:
            assert len(omega) == len(features_idx), "Features and weights different dimensionality"
            reward = (reward*omega).sum(axis=-1)
            discounted_phi.append(reward * gamma)
        else:
            discounted_phi.append(reward * np.tile(gamma, (r_dim, 1)).transpose())
    discounted_phi = np.array(discounted_phi)
    expected_discounted_phi = discounted_phi.sum(axis=0).sum(axis=0) / num_episodes
    print('Featrues Expectations:', expected_discounted_phi)
    # normalization factor computation
    if normalize_f:
        discounted_phi = np.array(discounted_phi)
        expected_discounted_phi = discounted_phi.sum(axis=0).sum(axis=0) / num_episodes
        if verbose:
            print('Expected discounted phi = ', expected_discounted_phi)
            input()
        #print('Expected discounted phi = ', expected_discounted_phi)
        #input()
        logger['expected_discounted_phi'] = expected_discounted_phi
        expected_discounted_phi = np.tile(expected_discounted_phi, (num_episodes, episode_length, 1))
        discounted_phi /= expected_discounted_phi

    # computing the gradients of the logarithm of the policy wrt policy parameters
    episode_gradients = []
    probs = []

    for step in range(steps):

        step_layers = pi.compute_gradients([x_dataset[step]], [y_dataset[step]])
        step_gradients = []
        for layer in step_layers:
            step_gradients.append(layer.ravel())
        step_gradients = np.concatenate(step_gradients)

        episode_gradients.append(step_gradients)
        if behavioral_pi is not None:
            target_pi_prob = pi.prob(x_dataset[step], y_dataset[step])
            behavioral_pi_prob = behavioral_pi.prob(x_dataset[step], y_dataset[step])
            probs.append(target_pi_prob / (behavioral_pi_prob + 1e-10))

            #print("Step:",step)
    gradients = []
    ratios = []

    for episode in range(num_episodes):
        base = episode * episode_length
        gradients.append(episode_gradients[base: base + episode_length])
        if behavioral_pi is not None:
            ratios.append(probs[base: base + episode_length])

    gradients = np.array(gradients)

    if behavioral_pi is not None:
        ratios = np.array(ratios)
    # GPOMDP optimal baseline computation
    num_params = gradients.shape[2]
    logger['num_params'] = num_params
    if omega is None:
        cum_gradients = np.transpose(np.tile(gradients, (r_dim, 1, 1, 1)), axes=[1, 2, 3, 0]).cumsum(axis=1)
        if behavioral_pi is not None:
            importance_weight = ratios.cumprod(axis=1)
            cum_gradients = cum_gradients * importance_weight
        phi = np.transpose(np.tile(discounted_phi, (num_params, 1, 1, 1)), axes=[1, 2, 0, 3])

    else:
        cum_gradients = gradients.cumsum(axis=1)
        if behavioral_pi is not None:
            importance_weight = ratios.cumprod(axis=1)
            cum_gradients = cum_gradients * importance_weight
        phi = np.transpose(np.tile(discounted_phi, (num_params, 1, 1)), axes=[1, 2, 0])


    '''
    # Freeing memory
    del X_dataset
    del y_dataset
    del r_dataset
    del episode_gradients
    del gamma
    del discounted_phi
    '''
    # GPOMDP objective function gradient estimation
    if use_baseline:
        num = (cum_gradients ** 2 * phi).sum(axis=0)
        den = (cum_gradients ** 2).sum(axis=0) + 1e-10
        baseline = num / den
        if omega is None:
            baseline = np.tile(baseline, (num_episodes, 1, 1, 1))
        else:
            baseline = np.tile(baseline, (num_episodes, 1, 1))
        if use_mask and dones_dataset is not None:
            mask = np.array(dones_dataset).reshape((num_episodes, episode_length))
            for ep in range(num_episodes):
                for step in range(episode_length):
                    if mask[ep, step] == 1.:
                        break
                    mask[ep, step] = 1
            if omega is None:
                baseline *= np.tile(mask, (num_params, r_dim, 1, 1)).transpose((2, 3, 0, 1))
            else:
                baseline *= np.tile(mask, (num_params, 1, 1)).transpose((1, 2, 0))

        phi = phi - baseline

    estimated_gradients = (cum_gradients * (phi)).sum(axis=1).mean(axis=0)

    return estimated_gradients, {'logger': logger}


def learn(*,
          network,
          env,
          eval_policy,
          total_timesteps,
          timesteps_per_batch=1024,  # what to train on
          gamma=0.9999,
          seed=None,
          max_episodes=0,
          max_iters=0,   # time constraint
          horizon=600,
          callback=None,
          load_path=None,
          checkpoint_path_in=None,
          checkpoint_dir_out=None,
          checkpoint_freq=100,  # In iterations!!,
          from_iter=0,
          eval_episodes=20,
          step_size=1e-3,
          step_size_strategy='adam',
          planner=None,
          sampler=None,
          eval_sampler=None,
          verbose=False,
          ac_space=None,
          pos_actions=False,
          tanh=False,
          beta=True,
          init_logstd=0.0,
          init_bias=0.0,
          **network_kwargs
          ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''


    # nworkers = MPI.COMM_WORLD.Get_size()
    # rank = MPI.COMM_WORLD.Get_rank()
    #
    # cpus_per_worker = 1
    # config = tf.ConfigProto(
    #     allow_soft_placement=True,
    #     inter_op_parallelism_threads=cpus_per_worker,
    #     intra_op_parallelism_threads=cpus_per_worker)
    # config.gpu_options.allow_growth = True
    # U.get_session(config=config)
    ob_space = env.observation_space
    if ac_space is None:
        ac_space = env.action_space

    policy = build_policy(ob_space=ob_space, ac_space=ac_space, policy_network=network, value_network='copy',
                          pos_actions=pos_actions, tanh=tanh, init_logstd=init_logstd, init_bias=init_bias,
                          **network_kwargs, beta=beta)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)

    #Loading checkpoint
    if checkpoint_path_in is not None and os.path.isfile(checkpoint_path_in):
        pi.load(checkpoint_path_in)
        logger.log('Loaded policy weights from %s' % checkpoint_path_in)


    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)

    U.initialize()
    if load_path is not None:
        pi.load(load_path)
    scores = []
    th_init = get_flat()
    adam_mean = Adam(len(th_init))
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    if sampler is None:
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    iters_eval = 0
    all_logs = []
    best_rew = -np.inf

    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        if iters_so_far % checkpoint_freq == 0 and iters_so_far != 0 and checkpoint_dir_out is not None:
            if not os.path.exists(checkpoint_dir_out):
                os.makedirs(checkpoint_dir_out)
            pi.save(os.path.join(checkpoint_dir_out, 'checkpoint_%d' % iters_so_far))
            logger.log('Saved policy weights as %s' % os.path.join(checkpoint_dir_out, 'checkpoint_%d.npy' % iters_so_far))
            if eval_sampler is not None:
                rewards, _, _ = eval_sampler.collect(get_flat(), planner.get_params())
                eval_sampler.restart()
                rew = np.mean(rewards)
            else:
                def pi_wrapper(ob):
                    ac, vpred, _, _, _ = pi.step(ob, stochastic=False)
                    return ac

                rew, _, logs = eval_policy(pi=pi_wrapper, n_episodes=eval_episodes, verbose=verbose)
                for log in logs:
                    log['iter'] = iters_eval
                all_logs = all_logs + logs

                keys = all_logs[0].keys()
                del_perc, del_act = env.get_delays()
                start_time = env.get_start_time()
                directory_output = env.get_dir_out()
                with open(directory_output + '/logs/eval_progress-' + str(start_time) + '.csv', 'w') as output_file:
                    dict_writer = csv.DictWriter(output_file, keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(all_logs)
            if rew > best_rew and checkpoint_dir_out is not None:
                pi.save(os.path.join(checkpoint_dir_out, 'best'))
                logger.log('Saved best policy weights as %s' % os.path.join(checkpoint_dir_out, 'best'))
                best_rew = rew

            iters_eval += 1
        print(colorize("done in %.3f seconds" % (time.time() - tstart), color='magenta'))
        if sampler is None:
                seg = seg_gen.__next__()
        else:
            seg = sampler.collect(get_flat(), planner.get_params())
            if iters_so_far % checkpoint_freq == 0:
                sampler.restart()

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, dones, rewards = seg["ob"], seg["ac"], seg["new"], seg["rew"]

        grad, _ = compute_gradient(pi=pi, x_dataset=ob.tolist(), y_dataset=ac.tolist(),
                                r_dataset=rewards.reshape(-1,1).tolist(), dones_dataset=dones.tolist(),
                                episode_length=horizon, use_baseline=True, use_mask=True, discount_f=gamma)
        if np.allclose(grad, 0):
            logger.log("Got zero gradient. not updating")
        elif not  np.isfinite(grad).all():
            logger.log("Nan Gradients! skipping update!!")
        else:
            grad2norm = np.linalg.norm(grad, 2)
            gradmaxnorm = np.linalg.norm(grad, np.infty)

            step_size_it = {'const': step_size,
                            'norm': step_size / grad2norm if grad2norm > 0 else 0,
                            'vanish': step_size / np.sqrt(iters_so_far + 1),
                            'adam': adam_mean.update(grad[:, 0], step_size)
                            }.get(step_size_strategy, step_size)

            if step_size_strategy == 'adam':
                delta_th = step_size_it
            else:
                delta_th = step_size_it * grad

            #pol.set_params(rho + delta_rho)
            thbefore = get_flat()
            thnew = thbefore + delta_th
            set_from_flat(thnew)

        lens, rews  = seg["ep_lens"], seg["ep_rets"]
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        scores.append((np.min(rews), np.max(rews), np.mean(rews)))
        np.save(checkpoint_dir_out + '/online_scores.npy', scores)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        ep_rew_mean = np.mean(rewbuffer)

        # Saving best
        if iters_so_far % checkpoint_freq == 0 and ep_rew_mean > best_rew and checkpoint_dir_out is not None:
            pi.save(os.path.join(checkpoint_dir_out, 'best'))
            best_rew = ep_rew_mean
            logger.log('Saved policy weights as %s' % os.path.join(checkpoint_dir_out, 'best.npy'))
        logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]