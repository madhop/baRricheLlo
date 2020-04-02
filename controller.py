from fqi.reward_function import *
from fqi.utils import *
from gym_torcs_ctrl import TorcsEnv

import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib


class Projection(Reward_function):
    def __init__(self, ref_t, clip_range, ref_dt, sample_dt, penalty=None):
        super().__init__(ref_t, clip_range, ref_dt, sample_dt, penalty)
        self.kdtree = KDTree(self.ref_p)
        self.seg = np.roll(self.ref_p, -1, axis=0) - self.ref_p  # The vector of the segment starting at each ref point
        self.seg_len = np.linalg.norm(self.seg, axis=1)  # The length of the segments
        self.cumul_len = np.cumsum(self.seg_len) - self.seg_len  # The cumulative length from the start to the ref i
        self.full_len = self.cumul_len[-1] + self.seg_len[-1]  # The full length of the ref trajectory

    # Compute how much time (in ref_dts) the reference would take to go from two consecutive states projections
    def _compute_ref_time(self, projection):
        ref_id, delta = projection
        t = ref_id[1:] + delta[1:] - (ref_id[:-1] + delta[:-1])
        t[t < -self.ref_len / 2] += self.ref_len  # if the next step is behind the current point
        return t

    # compute the distance over the reference of two consecutive state projections
    def _compute_ref_distance(self, projection):
        ref_id, delta = projection
        partial_len_next = delta[1:] * self.seg_len[ref_id[1:]]
        partial_len_curr = delta[:-1] * self.seg_len[ref_id[:-1]]
        d = self.cumul_len[ref_id[1:]] + partial_len_next - (self.cumul_len[ref_id[:-1]] + partial_len_curr)
        d[d < -self.full_len / 2] += self.full_len
        return d

    '''
    Compute the minimum distance projection of each state over the reference as a couple (ref_id, delta)
    Projects the state on both the segments connected to the closest reference point and then select the closest projection
    '''

    def _compute_projection(self, state):
        _, ref_id = self.kdtree.query(state)
        ref_id = ref_id.squeeze()  # index of the next segment
        prev_ref_id = ref_id - 1  # index of the previous segment
        # prev_ref_id[prev_ref_id == -1] = self.ref_len - 1  # if we got before the ref point 0
        ref_state = state - self.ref_p[ref_id]  # vector from the ref point to the state
        prev_ref_state = state - self.ref_p[prev_ref_id]
        delta = np.sum(self.seg[ref_id] * ref_state, axis=1) / np.square(self.seg_len[ref_id])  # <s-r,seg>/|seg|^2
        delta_prev = np.sum(self.seg[prev_ref_id] * prev_ref_state, axis=1) / np.square(self.seg_len[prev_ref_id])
        delta = delta.clip(0, 1);  # clips to points within the segment
        delta_prev = delta_prev.clip(0, 1);
        dist = np.linalg.norm(state - (self.ref_p[ref_id] + delta[:, None] * self.seg[ref_id]),
                              axis=1)  # point-segment distance
        dist_prev = np.linalg.norm(state - (self.ref_p[prev_ref_id] + delta_prev[:, None] * self.seg[prev_ref_id]),
                                   axis=1)
        closest = np.argmin(np.column_stack((dist, dist_prev)), axis=1)  # index of the one with minimum distance
        delta = np.column_stack((delta, delta_prev))[
            np.arange(delta.shape[0]), closest]  # select the one with minimum distance
        ref_id = np.column_stack((ref_id, prev_ref_id))[np.arange(1), closest]
        return ref_id, delta


class MeanController(object):
    def __init__(self, ref_df, env=None, alpha1=0.5, alpha2=0.02, speed_y_thr=5, beta1=0.055, gamma1=3, gamma2=73.5, gamma3=116, k=20):
        # Init
        self.env = env
        # Throttle params
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.speed_y_thr = speed_y_thr
        # Break params
        self.beta1 = beta1
        # Steering params
        self.gamma1 = gamma1  # rho param
        self.gamma2 = gamma2  # orientation parm
        self.gamma3 = gamma3  # angle param

        self.k = k

        self.ref_df = ref_df
        self.projector = Projection(ref_t=self.ref_df, clip_range=None, ref_dt=1, sample_dt=10, penalty=None)

        self.previous_steer = 0
        self.prev_rho = 0
        self.prev_theta = 0
        self.prev_ref_theta = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def yaw_diff(self, x, y):
        x = x.copy()
        y = y.copy()
        if abs(x) > np.pi / 2 and abs(y) > np.pi / 2:
            if x < 0:
                x = 2 * np.pi + x
            if y < 0:
                y = 2 * np.pi + y
        return x - y

    def yaw_proj(self, x0, x1, delta):
        x0 = x0.copy()
        x1 = x1.copy()
        x0_sign = 1 if x0 >= 0 else -1
        x1_sign = 1 if x1 >= 0 else -1
        if abs(x0) > np.pi / 2 and abs(x1) > np.pi / 2 and ((x0_sign * x1_sign) < 0):
            if x0 < 0:
                x0 = 2 * np.pi + x0
            if x1 < 0:
                x1 = 2 * np.pi + x1
        return (1 - delta) * x0 + delta * x1

    def forward_ref_id(self, ref_id, k):
        if ref_id + k < self.ref_df.shape[0]:
            return ref_id + k
        else:
            return ref_id + k - self.ref_df.shape[0]

    def act(self, obs, verbose=True):
        # Find projection on reference trajectory
        state_p = np.array([[obs['x'], obs['y']]])
        ref_id, delta = self.projector._compute_projection(state_p)
        ref_id = ref_id[0]
        delta = delta[0]

        # RHO component
        # compute rho as delta_trackPos
        trackPoss_proj = ((1 - delta) * self.ref_df['trackPos'].values[ref_id] +
                          delta * self.ref_df['trackPos'].values[self.forward_ref_id(ref_id, 1)])

        rho = (trackPoss_proj - obs['trackPos']) / 2

        # Speed component
        Vref_proj = (1 - delta) * self.ref_df['speed_x'].values[ref_id] +\
                    delta * self.ref_df['speed_x'].values[self.forward_ref_id(ref_id, 1)]
        V = obs['speed_x']

        # Yaw components
        ref_O = self.ref_df['yaw'].values[ref_id]
        ref_O1 = self.ref_df['yaw'].values[self.forward_ref_id(ref_id, 1)]
        # ref_O_proj = (1 - delta) * ref_O + delta * ref_O1
        ref_O_proj = self.yaw_proj(ref_O, ref_O1, delta)

        delta_O = self.yaw_diff(ref_O_proj, obs['yaw'])
        delta_O = delta_O / (2 * np.pi)

        delta_ref_O = self.yaw_diff(self.ref_df['yaw'].values[self.forward_ref_id(ref_id, self.k)], ref_O)
        delta_ref_O = delta_ref_O / (2 * np.pi)  # scale from -1 to 1

        # Compute steer
        c_rho = np.tanh(self.gamma1 * rho)
        c_theta = np.tanh(self.gamma2 * delta_O)
        c_ref_theta = np.tanh(self.gamma3 * delta_ref_O)

        c_steer = self.previous_steer
        steer = np.dot(np.ones(3, ) / 4, np.array([c_rho - self.prev_rho,
                                                   c_theta - self.prev_theta,
                                                   c_ref_theta - self.prev_ref_theta])) + c_steer
        if verbose:
            print('-------------------------')
            #print('STEER {:.4f} + {:.4f} + {:.4f} + {:.4f}'.format(self.weights[0] / 4 * (c_rho - self.prev_rho),
            #                                                       self.weights[1] / 4 * (c_theta - self.prev_theta),
            #                                                       self.weights[2] / 4 * (c_ref_theta - self.prev_ref_theta),
            #                                                       c_steer))

        self.previous_steer = steer
        self.prev_rho = c_rho
        self.prev_theta = c_theta
        self.prev_ref_theta = c_ref_theta
        if verbose:
            print('REF STEER {:.4f}'.format(self.ref_df['Steer'].values[ref_id]))
            print('CAR STEER {:.4f}'.format(steer))
            print('RHO {:.4f} DELTA O {:.4f} DELTA REF O {:.4f}'.format(rho, delta_O, delta_ref_O))

        #print('TPOS {:.4f} => {:.4f}<{:.4f}>{:.4f} => RHO {:.4f}'.format(obs['trackPos'], rb, trackPoss_proj, lb, rho))

        brake = max(0, self.beta1 * (V - Vref_proj))
        brake = min(brake, 1)
        speed_y = abs(obs['speed_y'])
        if speed_y < 5:
            speed_y = 0
        throttle = min(1, self.alpha1 * (Vref_proj - V))
        throttle = max(throttle - self.alpha2 * speed_y, 0)

        if verbose:
            print('V - VR {:.4f}'.format(V - Vref_proj))
            print('CAR BRAKE {:.4f}'.format(brake))
            print('REF BRAKE: {:.4f}'.format(self.ref_df['Brake'].values[ref_id]))
            print('CAR THR {:.4f}'.format(throttle))
            print('REF THR: {:.4f}'.format(self.ref_df['Throttle'].values[ref_id]))
            print('SPEED X {:.4f} SPEED Y {:.4f}'.format(obs['speed_x'], obs['speed_y']))

        info = {'rho': rho, 'delta_O': delta_O, 'delta_ref_O': delta_ref_O, 'vr': Vref_proj, 'v': V,
                'steer_r': self.ref_df['Steer'].values[ref_id], 'ref_id': ref_id, 'delta': delta}
        return [steer, brake, throttle], info
    
    def action_closure(self, obs, params):
        #set params
        params = params()
        print('params:', params)
        self.alpha1 = params[0]
        self.alpha2 = params[1]
        self.speed_y_thr = params[2]
        # Break params
        self.beta1 = params[3]
        # Steering params
        self.gamma1 = params[4]  # rho param
        self.gamma2 = params[5]  # orientation parm
        self.gamma3 = params[6]  # angle param
        #act
        action, _ = self.act(obs, False)
        return action

    def playGame(self, episode_count=1, max_steps=100000, save_data=False):
        step = 0
        episode = list()
        for i in range(episode_count):
            if np.mod(i, 3) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = self.env.reset(relaunch=True)
                episode = list()
            else:
                ob = self.env.reset()
                episode = list()

            for j in range(max_steps):
                if j % 1 == 0:
                    action, info = self.act(ob, True)
                else:
                    action, info = self.act(ob, False)

                # print(action)
                ob['steer'] = action[0]
                ob['brake'] = action[1]
                ob['throttle'] = action[2]
                ob['rho'] = info['rho']
                ob['delta_O'] = info['delta_O']
                ob['delta_ref_O'] = info['delta_ref_O']
                ob['vr'] = info['vr']
                ob['v'] = info['v']
                ob['steer_r'] = info['steer_r']
                ob['ref_id'] = info['ref_id']
                ob['delta'] = info['delta']

                episode.append(ob)
                ob, reward, done, _ = self.env.step(action)
                step += 1
                if done:
                    break

        return episode
