#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:48:48 2020

@author: umberto
Driver controller
"""

import numpy as np
import pandas as pd
from fqi.reward_function import Reward_function

from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


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
        #prev_ref_id[prev_ref_id == -1] = self.ref_len - 1  # if we got before the ref point 0
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
        delta = np.column_stack((delta, delta_prev))[np.arange(delta.shape[0]), closest]  # select the one with minimum distance
        ref_id = np.column_stack((ref_id, prev_ref_id))[np.arange(1), closest]
        return ref_id, delta
#%%
class Controller():
    def __init__(self):
        # Init
        self.alpha1 = None
        self.k1 = None
        self.beta1 = None
        self.k2 = None
        self.gamma1 = None
        self.gamma2 = None
        self.ref_df = pd.read_csv('trajectory/ref_traj.csv')
        self.projector = Projection(ref_t=self.ref_df, clip_range=None, ref_dt=1, sample_dt=10, penalty=None)
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def act(self, obs):
        # Find projection on reference trajectory
        state_p = np.array([[obs['xCarWorld'], obs['yCarWorld']]])
        ref_id, delta = self.projector._compute_projection(state_p)
        Vref = self.ref_df['speed_x'].values[ref_id]
        V = obs['speed_x']
        r = np.array([self.ref_df['xCarWorld'].values[ref_id], self.ref_df['yCarWorld'].values[ref_id]])
        r1 = np.array([self.ref_df['xCarWorld'].values[ref_id+1], self.ref_df['yCarWorld'].values[ref_id+1]])
        ref_segment = (r1 - r) * delta
        ref_proj = (r + ref_segment).reshape(1,2)
        
        
        p = ref_proj - state_p    # position of the car wrt the reference
        """delta_O = None  # delta orientation of the car
        # Compute actions
        throttle = self.sigmoid(self.alpha1 * (Vref - V) + self.k1 * np.power(V, 2))
        brake = self.sigmoid(self.beta1 * (V - Vref) + self.k2 * np.power(V, 2))
        steer = np.tanh(self.gamma1 * p + self.gamma2 * delta_O)
        return [steer, brake, throttle]"""
        
    
    
    

if __name__ == '__main__':
    C = Controller()
    
    
#%%
data = pd.read_csv('trajectory/dataset_human.csv')
state_p = data.head(1)[['xCarWorld', 'yCarWorld', 'actualSpeedModule']].values
C.projector._compute_projection(state_p)

#%%
obs  = {'xCarWorld': 654.799744, 'yCarWorld': 1169.202148, 'speed_x': 315.11856326}
C.act(obs)

#%%
import matplotlib.pyplot as plt
ref_dt = pd.read_csv('trajectory/ref_traj.csv')
plt.scatter(ref_dt[99:101].xCarWorld, ref_dt[99:101].yCarWorld)
plt.scatter(data[8:9].xCarWorld, data[8:9].yCarWorld)
plt.show()
