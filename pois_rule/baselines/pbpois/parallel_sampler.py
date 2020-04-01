from multiprocessing import Process, Queue, Event
import os
import baselines.common.tf_util as U
import time
import sys
from mpi4py import MPI
from baselines.common import set_global_seeds as set_all_seeds
import numpy as np
import tensorflow as tf
def traj_segment_function(pol, env, gamma, task_horizon, feature_fun, num_episodes=1):
    rets = []
    disc_rets = []
    lens = []
    for i in range(num_episodes):
        ret = disc_ret = 0
        t = 0

        ob = env.reset()
        done = False
        while not done and t < task_horizon:
            s = feature_fun(ob) if feature_fun else ob
            a = pol.act(s)
            ob, r, done, _ = env.step(a)
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

    def __init__(self, output, input, event, make_env, make_pi, traj_segment_generator, seed,index):
        super(Worker, self).__init__()
        self.output = output
        self.input = input
        self.make_env = make_env
        self.make_pi = make_pi
        self.traj_segment_generator = traj_segment_generator
        self.event = event
        self.seed = seed
        self.index=index

    def close_env(self):
        self.env.close()

    def run(self):

        sess = U.single_threaded_session()
        sess.__enter__()

        env = self.make_env(self.index)
        self.env=env
        env.reset()
        workerseed = self.seed + 10000 * MPI.COMM_WORLD.Get_rank()
        #env.seed(workerseed)
        pi = self.make_pi("pi_"+str(self.index))
        np.random.seed(workerseed)
        print('Worker %s - Running with seed %s' % (os.getpid(), workerseed))

        while True:
            self.event.wait()
            self.event.clear()
            command, weights = self.input.get()
            if command == 'collect':
                pi.set_theta(weights)
                #print('Worker %s - Collecting...' % os.getpid())
                samples = self.traj_segment_generator(pi, env)
                #print('Worker %s - Collected...' % os.getpid())
                self.output.put((os.getpid(), samples))
            elif command == 'exit':
                print('Worker %s - Exiting...' % os.getpid())
                env.close()
                sess.close()
                break

class ParallelSampler(object):

    def __init__(self, make_pi, make_env,n_workers, horizon, feature_fun, episodes_per_worker=1, gamma=0.999, seed=0):
        self.n_workers=n_workers

        print('Using %s CPUs' % self.n_workers)

        if seed is None:
            seed = time.time()

        self.output_queue = Queue()
        self.input_queues = [Queue() for _ in range(self.n_workers)]
        self.events = [Event() for _ in range(self.n_workers)]

        n_episodes=n_workers
        n_episodes_per_process = episodes_per_worker
        print("%s episodes per worker" %n_episodes_per_process)

        f = lambda pi, env: traj_segment_function(pi, env, gamma, horizon, feature_fun, num_episodes=episodes_per_worker)
        fun = [f] * (self.n_workers)
        self.workers = [Worker(self.output_queue, self.input_queues[i], self.events[i], make_env, make_pi, fun[i], seed + i,i) for i in range(self.n_workers)]

        for w in self.workers:
            w.start()


    def collect(self, actor_weights):
        for i in range(self.n_workers):
            self.input_queues[i].put(('collect', actor_weights))

        for e in self.events:
            e.set()

        sample_batches = []
        for i in range(self.n_workers):
            pid, samples = self.output_queue.get()
            sample_batches.append(samples)

        return self._merge_sample_batches(sample_batches)

    def _merge_sample_batches(self, sample_batches):

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


def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]