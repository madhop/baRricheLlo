from multiprocessing import Process, Queue, Event
import os
import baselines.common.tf_util as U
import time
from mpi4py import MPI
from baselines.common import set_global_seeds as set_all_seeds
import numpy as np
import tensorflow as tf

def traj_segment_function(pi, env, gamma, n_episodes, horizon, stochastic):
    '''
    Collects trajectories
    '''

    # Initialize state variables
    t = 0

    #ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()
    ac, _, _, _, _ = pi.step(ob, stochastic=True)
    #ac = ac.flatten
    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon * n_episodes)])
    rews = np.zeros(horizon * n_episodes, 'float32')
    news = np.zeros(horizon * n_episodes, 'int32')
    acs = np.array([ac for _ in range(horizon * n_episodes)])

    i = 0
    j = 0
    while True:
        prevac = ac
        ac, vpred, _, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if i == n_episodes:
            return {"ob": obs, "rew": rews,  "new": news, "ac": acs, "ep_rets": ep_rets, "ep_lens": ep_lens}

        obs[j + i*horizon] = ob
        news[j + i * horizon] = new
        acs[j + i * horizon] = ac

        ob, rew, new, _ = env.step(ac)
        rews[j + i * horizon] = rew

        #print("Step %s" %(j))
        #rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        j += 1
        if new or j == horizon:
            new = True
            env.done = True

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)

            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            i += 1
            j = 0
        t += 1

def eval_traj_segment_function(pi, env, gamma, n_episodes, horizon, stochastic):
    rets = []
    disc_rets = []
    lens = []
    for i in range(n_episodes):
        ret = disc_ret = 0
        t = 0
        ob = env.reset()
        done = False
        while not done and t < horizon:
            ac, _, _, _, _ = pi.step(ob, stochastic=stochastic)
            ob, r, done, _ = env.step(ac)
            ret += r
            disc_ret += gamma ** t * r
            t += 1
        rets.append(ret)
        disc_rets.append(disc_ret)
        lens.append(t)
    return {"ret": rets, "disc_ret": disc_rets, "len": lens}

class Worker(Process):
    '''
    A worker is an independent process with its own environment and policy instantiated locally
    after being created. It ***must*** be runned before creating any tensorflow session!
    '''

    def __init__(self, output, input, event, make_env, make_pi, make_planner, traj_segment_generator, seed,index):
        super(Worker, self).__init__()
        self.output = output
        self.input = input
        self.make_env = make_env
        self.make_pi = make_pi
        self.make_planner = make_planner
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed
        self.index=index

    def close_env(self):
        self.env.close()

    def run(self):
        import tensorflow as tf
        with tf.Session(config=tf.ConfigProto(use_per_session_threads=True)) as sess:
            planner = None
            if self.make_planner is not None:
                self.planner = planner = self.make_planner()
            self.env = env = self.make_env(self.index, planner)
            env.reset()
            workerseed = self.seed + 10000 * MPI.COMM_WORLD.Get_rank()
            set_all_seeds(workerseed)
            env.seed = workerseed
            scope = 'pi_%s' % os.getpid()
            pi = self.make_pi(scope, sess)
            var = get_pi_trainable_variables(scope)
            set_from_flat = U.SetFromFlat(var)
            print('Worker %s - Running with seed %s' % (os.getpid(), workerseed))

            while True:
                self.event.wait()
                self.event.clear()
                command, args = self.input.get()
                if command == 'collect':
                    weights = args["actor_weights"]
                    planner_weights = args["planner_weights"]
                    if self.planner is not None and planner_weights is not None:
                        self.planner.set_params(planner_weights)
                    set_from_flat(weights)
                    print('Worker %s - Collecting...' % os.getpid())
                    samples = self.traj_segment_generator(pi, env, self.planner)
                    self.output.put((os.getpid(), samples))
                elif command == 'exit':
                    print('Worker %s - Exiting...' % os.getpid())
                    env.close()
                    break


class ParallelSampler(object):

    def __init__(self, make_pi, make_env, make_planner, n_workers, stochastic, horizon, episodes_per_worker=1,
                 gamma=0.999, seed=0, eval=False):
        self.n_workers = n_workers

        print('Using %s CPUs' % self.n_workers)

        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]
        self.seed = seed
        self.make_env = make_env
        self.make_pi = make_pi
        self.make_planner = make_planner
        self.eval = eval
        n_episodes_per_process = episodes_per_worker
        print("%s episodes per worker" %n_episodes_per_process)
        if eval:
            collect_function = eval_traj_segment_function
        else:
            collect_function = traj_segment_function
        self.collect_function = collect_function
        f = lambda pi, env, planner: collect_function(pi, env, gamma, n_episodes_per_process, horizon,  stochastic)
        self.fun = fun = [f] * (self.n_workers)
        self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i], make_env, make_pi,
                               make_planner, fun[i], seed + i, i) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()

    def collect(self, actor_weights, planner_weights=None):
        args = {'actor_weights': actor_weights, 'planner_weights': planner_weights}
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', args))

        for e in self.events:
            e.set()

        sample_batches = []
        for i in range(self.n_workers):
            pid, samples = self.output_queue.get()
            sample_batches.append(samples)
        if self.eval:
            return self._merge_sample_batches_eval(sample_batches)
        else:
            return self._merge_sample_batches(sample_batches)

    def _merge_sample_batches(self, sample_batches):
        '''
        {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
         "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
         "ep_rets": ep_rets, "ep_lens": ep_lens}
         '''
        np_fields = ['ob', 'rew', 'new', 'ac']
        list_fields = ['ep_rets', 'ep_lens']

        new_dict = list(zip(np_fields, map(lambda f: sample_batches[0][f], np_fields))) + \
                   list(zip(list_fields,map(lambda f: sample_batches[0][f], list_fields)))

        new_dict = dict(new_dict)

        for batch in sample_batches[1:]:
            for f in np_fields:
                new_dict[f] = np.concatenate((new_dict[f], batch[f]))
            for f in list_fields:
                new_dict[f].extend(batch[f])
        return new_dict

    def _merge_sample_batches_eval(self, sample_batches):
        rets = []
        disc_rets = []
        lens = []
        for batch in sample_batches:
            rets += batch["ret"]
            disc_rets += batch["disc_ret"]
            lens += batch["len"]
        return rets, disc_rets, lens

    def close(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        for e in self.events:
            e.set()
        for w in self.workers:
            #w.close_env()
            w.join()

    def restart(self):
        for i in range(self.n_workers):
            self.input_queues[i].put(('exit', None))

        # Set the internal flag to true. All threads waiting for it to become true are awakened.
        # Threads that call wait() once the flag is true will not block at all
        for e in self.events:
            e.set()

        for w in self.workers:
            #w.close_env()
            w.join()

        for w in self.workers:
            #w.close_env()
            w.terminate()
            del w

        self.workers = [
            Worker(self.output_queue, self.input_queues[i],  self.events[i], self.make_env, self.make_pi,
                   self.make_planner, self.fun[i], self.seed + i, i) for i in range(self.n_workers)]
        for w in self.workers:
            w.start()


def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]