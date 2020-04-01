import numpy as np
import time


def eval_policy(pi, n_episodes=100, env=None, env_sampler=None, network="round_robin", add_terminal=False, verbose=True, interactive=False,
                print_states=False):

    rewards = []
    logs = []
    if env is not None:
        for i in range(n_episodes):

            start = time.time()
            s = env.reset(network=network)
            t = 0
            rew = 0
            if print_states:
                env.print_state(s)
            while True:
                s = np.concatenate([s, [0]]) if add_terminal else s
                a = pi(s)
                ns, r, done, inf = env.step(a)
                s = ns
                if interactive:
                    #print("Action=%f" % a.flatten())
                    print("Reward=%f" % r)
                    input()
                rew += r
                t += 1
                if print_states:
                    env.print_state(ns)
                    if inf['crash']:
                        print("has crashed!!")
                        input()
                if done:
                    break

            if verbose:
                print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
            rewards.append(rew)
            logs.append(env._get_info())
    elif env_sampler is not None:
        pi.set_mean()
        theta = pi.get_theta()
        rewards, _, _ = env_sampler.collect(theta)
    else:
        raise ValueError("Specify an environment to sample!")

    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))
    if env is not None:
        env.reset()

    return avg, std, logs
