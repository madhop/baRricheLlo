import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf

def plot_probs(ax,logits, executed_actions):
    ax.clear()
    probs = ax.bar(index, logits, bar_width,
                    alpha=0.4, color='b')
    probs[executed_actions].set_color('r')
    ax.set_xlabel('Action')
    ax.set_ylabel('Probability')
    ax.set_title('Policy')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(action_labels)
    ax.set_ylim(bottom=0, top=1)

    plt.show()
    return probs


g = { "barlist":None}
num_actions = 3
action_labels = ["NOP", "CSD", "HSD"]
plt.ion()
fig, ax = plt.subplots()
index = np.arange(num_actions)
bar_width = 0.35
g["barlist"] = plot_probs(ax, [0.33, 0.33, 0.33], 0)
#input()


def eval_policy(env, pi, n_episodes, network="round_robin", add_terminal=False, verbose=True, interactive=False,
                print_states=False):

    rewards = []
    logs = []
    probs=[[] for _ in range(num_actions)]
    probs.append([]) #forbidden

    for i in range(n_episodes):

        start = time.time()
        s = env.reset(network=network)
        t = 0
        rew = 0
        forbidden_count = 0
        while True:
            s = np.concatenate([s, [0]]) if add_terminal else s
            a, logits = pi(s)
            a = a[0]

            #plot_probs(ax, logits[0], a)
            g["barlist"] = update_plot(g["barlist"], ax, logits[0], a)
            #if s[0] and s[1]:
                #input()
            ns, r, done, inf = env.step(a)
            s = ns
            if interactive:
                print("action=%f" %(a))
                print("Reward=%f" %(r))
                input()
            rew += r
            t += 1
            env.print_state(ns, inf)
            if done:

                break

        if verbose:
            print("Episode {0}: Return = {1}, Forbidden Actions= {2} Duration = {3}, Time = {4} s".format(i, rew,
                                                                                                          forbidden_count , t, time.time() - start))
        rewards.append(rew)
        logs.append(env._get_info())

        if rew < -600:
            print("Crash! press a key")
            #input()

    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))

    env.reset()

    return avg, std, logs


def update_plot(barlist,ax,logits,a):
    barlist.remove()
    barlist = ax.bar(index, logits, bar_width,
                    alpha=0.4, color='b')
    barlist[a].set_color('r')
    plt.draw()
    plt.pause(0.01)
    return barlist
