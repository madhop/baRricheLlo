#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:48:48 2020

@author: umberto
Driver controller
"""
from fqi.reward_function import *
from fqi.utils import *
from gym_torcs_ctrl import TorcsEnv

import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


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
#%% Define Controller
class Controller():
    def __init__(self, env, alpha1=1, k1=1, beta1=1, k2=1, gamma1=1, gamma2=1, gamma3=1, Kp=1, Ki=1, Kd=1):
        # Init
        self.env = env
        # Throttle params
        self.alpha1 = alpha1
        self.k1 = k1
        # Break params
        self.beta1 = beta1
        self.k2 = k2
        # Steering params
        self.gamma1 = gamma1  # rho param
        self.gamma2 = gamma2  # orientation parma
        self.gamma3 = gamma3  # steering maintenance
        self.ref_df = pd.read_csv('trajectory/ref_traj_yaw.csv')
        self.projector = Projection(ref_t=self.ref_df, clip_range=None, ref_dt=1, sample_dt=10, penalty=None)
        
        # steering PID 
        self.previous_error = None
        self.integral = 0
        self.dt = 0.05
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def steer_rho_PID(self, rho):
        #loop:
        error = rho#setpoint − measured_value
        self.integral = self.integral + error * self.dt
        if self.previous_error is not None:
            self.derivative = (error - self.previous_error)/self.dt
        else:
            self.derivative = 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * self.derivative
        self.previous_error = error
            
        return np.clip(output, -1,1)
    
    
    def act(self, obs):
        th_rho = 0#0.3
        th_O = 0.01
        # Find projection on reference trajectory
        state_p = np.array([[obs['x'], obs['y']]])
        ref_id, delta = self.projector._compute_projection(state_p)
        # compute rho as delta_trackPos
        """r = np.array([self.ref_df['xCarWorld'].values[ref_id], self.ref_df['yCarWorld'].values[ref_id]])
        r1 = np.array([self.ref_df['xCarWorld'].values[ref_id+1], self.ref_df['yCarWorld'].values[ref_id+1]])
        ref_segment = (r1 - r) * delta
        ref_proj = (r + ref_segment).reshape(1,2)
        rho = ref_proj - state_p
        rho = np.linalg.norm(rho) if rho[0][1] > 0 else -np.linalg.norm(rho)    # position of the car wrt the reference"""
        trackPoss_proj = (1-delta)*self.ref_df['trackPos'].values[ref_id]+delta*self.ref_df['trackPos'].values[ref_id+1]
        rho = (trackPoss_proj - obs['trackPos'])/2  # scale from -1 to 1
        rho = rho if np.absolute(rho) > th_rho else 0   # tollerance near the ref traj
        #rho = self.steer_rho_PID(rho)
        # compute velocity
        Vref_proj = (1-delta)*self.ref_df['speed_x'].values[ref_id]+delta*self.ref_df['speed_x'].values[ref_id+1]
        V = obs['speed_x']
        
        ref_O = self.ref_df['nYawBody'].values[ref_id]
        ref_O1 = self.ref_df['nYawBody'].values[ref_id+1]
        delta_O = ((1-delta)*self.ref_df['nYawBody'].values[ref_id]+delta*self.ref_df['nYawBody'].values[ref_id+1]) - obs['yaw']
        #delta_O = ref_O - obs['yaw'] # delta orientation of the car
        delta_O = delta_O/(2*np.pi) # scale from -1 to 1
        delta_ref_O = ref_O1 - ref_O
        delta_ref_O = delta_ref_O/(2*np.pi) # scale from -1 to 1
        
        # Compute actions
        l = ['rho', 'delta_O', 'delta_ref_O']
        #print(rho, '-', l[np.argmax([np.abs(self.gamma1 * rho), np.abs(self.gamma2 * delta_O), np.abs(self.gamma3 * delta_ref_O)])])
        #steer = 0.75 * np.tanh(self.gamma1 * rho + self.gamma2 * delta_O + self.gamma3 * delta_ref_O)
        steer = self.steer_rho_PID(rho)
        brake = self.sigmoid(self.beta1 * (V - Vref_proj) + self.k2 * np.power(V, 2))
        throttle = self.sigmoid(self.alpha1 * (Vref_proj - V) + self.k1 * np.power(V, 2))
        return [steer, brake, throttle], rho, delta_O, delta_ref_O, self.ref_df['Steer'].values[ref_id], obs['x'], obs['y'], self.ref_df['xCarWorld'].values[ref_id], self.ref_df['yCarWorld'].values[ref_id]
    
    
    def playGame(self, episode_count=1, max_steps=100000, save_data=False):
        step = 0 
        for i in range(episode_count):
            if np.mod(i, 3) == 0:
                # Sometimes you need to relaunch TORCS because of the memory leak error
                ob = self.env.reset(relaunch=True)
            else:
                ob = self.env.reset()
                
            for j in range(max_steps):
                action, rho, delta_O, delta_ref_O = self.act(ob)
                #print(action)
                ob, reward, done, _ = self.env.step(action)
                
                step += 1
                if done:
                    break
                

#%% main
if __name__ == '__main__':
    ref_df = pd.read_csv('trajectory/ref_traj.csv')
    simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
    # Reward function
    penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
    right_laps = np.array([ 9., 14., 16., 17., 20., 47., 49., 55., 59., 60., 61., 62., 63., 65., 68.])
    penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
    reward_function = Temporal_projection(ref_df, penalty=penalty)
    env = TorcsEnv(reward_function,collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=0, slow=False, faster=False, graphic=True)
    gamma1=0.0025    #rho
    gamma2=0.1     #delta_O
    gamma3=24      #delta_ref_O
    alpha1=1
    k1=0.000001
    k2=0
    max_steps=100000
    C = Controller(env, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, alpha1=alpha1, k1=k1, k2=k2)
    C.playGame()
    
    
"""#%% Build env
ref_df = pd.read_csv('trajectory/ref_traj.csv')
simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 9., 14., 16., 17., 20., 47., 49., 55., 59., 60., 61., 62., 63., 65., 68.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)
env = TorcsEnv(reward_function,collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
           gear_change=False, brake=True, start_env=False, damage_th=0, slow=False, faster=False, graphic=True)


#%% play game and store data
gamma1=0.002    #rho
gamma2=(2*np.pi) * 0.1     #delta_O
gamma3=(2*np.pi) * 24      #delta_ref_O
Tu = 3.
Kp = 0.01
Ki = 0.001#1.2*(Kp/0.45)/Tu #0.2    
Kd = 0.01#(3*(Kp/0.45)*Tu)/40
alpha1=1
k1=0.000001
k2=0
max_steps=100000
C = Controller(env, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, alpha1=alpha1, k1=k1, k2=k2, Kp=Kp,Ki=Ki,Kd=Kd)
step=0
action_vars = {'rho':[], 'delta_O':[], 'delta_ref_O':[], 'ref_action':[], 'action':[], 'x':[], 'y':[], 'ref_x':[], 'ref_y':[], 'integral':[], 'derivative':[]}
ob = C.env.reset(relaunch=True)
for _ in range(max_steps):
    action, rho, delta_O, delta_ref_O, ref_action, x, y, ref_x, ref_y = C.act(ob)
    action_vars['action'].append(action)
    action_vars['rho'].append(rho)
    action_vars['delta_O'].append(delta_O)
    action_vars['delta_ref_O'].append(delta_ref_O)
    action_vars['ref_action'].append(ref_action)
    action_vars['x'].append(x)
    action_vars['y'].append(y)
    action_vars['ref_x'].append(ref_x)
    action_vars['ref_y'].append(ref_y)
    action_vars['integral'].append(C.integral)
    action_vars['derivative'].append(C.derivative)
    ob, reward, done, _ = C.env.step(action)
    
    step += 1
    if done:
        break
        
C.env.end()

#%% plot PID staff
fig, axs = plt.subplots(3, 1)
axs[0].set_title('Kp:'+str(Kp)+ ' Ki:'+str(Ki)+ ' Kd:'+str(Kd))
axs[0].plot(list(map(lambda x: x*Kp, action_vars['rho'])), label='rho')
axs[0].plot(list(map(lambda x: x*Ki, action_vars['integral'])), label='integral')
axs[0].plot(list(map(lambda x: x*Kd, action_vars['derivative'])), label='derivative')
#axs[0].plot(list(map(lambda x: x*gamma3, action_vars['delta_ref_O'])), label='delta_ref_O')
axs[0].grid(True)
axs[0].legend()

axs[1].plot([x[0][0] for x in action_vars['action']], label='action')
axs[1].plot(action_vars['ref_action'], label='ref action')
axs[1].grid(True)
axs[1].legend()

axs[2].scatter(action_vars['x'], action_vars['y'], label='car', s=0.5)
axs[2].scatter(action_vars['ref_x'], action_vars['ref_y'], label='ref', s=0.5)
axs[2].legend()
plt.show()
#%%
for i in range(len(action_vars['x'])):
    plt.plot([action_vars['x'][i], action_vars['ref_x'][i]], [action_vars['y'][i], action_vars['ref_y'][i]])
#plt.scatter(action_vars['x'], action_vars['y'], label='car', s=0.5)
#plt.scatter(action_vars['ref_x'], action_vars['ref_y'], label='ref', s=0.5)
#plt.legend()
plt.show()
#%% plot vars and steering action
fig, axs = plt.subplots(3, 1)
axs[0].plot(list(map(lambda x: x*gamma1, action_vars['rho'])), label='rho')
axs[0].plot(list(map(lambda x: x*gamma2, action_vars['delta_O'])), label='delta_O')
axs[0].plot(list(map(lambda x: x*gamma3, action_vars['delta_ref_O'])), label='delta_ref_O')
axs[0].grid(True)
axs[0].legend()

axs[1].plot([x[0][0] for x in action_vars['action']], label='action')
axs[1].plot(action_vars['ref_action'], label='ref action')
axs[1].grid(True)
axs[1].legend()

axs[2].scatter(action_vars['x'], action_vars['y'], label='car', s=0.5)
axs[2].scatter(action_vars['ref_x'], action_vars['ref_y'], label='ref', s=0.5)
axs[2].legend()
plt.show()
#%%
data = pd.read_csv('trajectory/dataset_human.csv')
state_p = data.head(1)[['xCarWorld', 'yCarWorld', 'actualSpeedModule', 'nYawBody']].values
#C.projector._compute_projection(state_p)

#%% check act function
obs  = {'xCarWorld': 654.799744, 'yCarWorld': 1169.202148, 'speed_x': 310.11856326, 'nYawBody': 0.013237}
action = C.act(obs)

#%% check projection
trackPoss = [0.721191, 0.708962, 0.695912]
p = [np.array([[836.23, 1173.43]]), np.array([[840.138, 1173.37]]), np.array([[844.053, 1173.29]])]
state_p = p[0]
ref_id, delta = C.projector._compute_projection(state_p)
r = np.array([C.ref_df['xCarWorld'].values[ref_id], C.ref_df['yCarWorld'].values[ref_id]])
r1 = np.array([C.ref_df['xCarWorld'].values[ref_id+1], C.ref_df['yCarWorld'].values[ref_id+1]])
ref_segment = (r1 - r) * delta
ref_proj = (r + ref_segment).reshape(1,2)

#%%
plt.scatter(state_p[0][0], state_p[0][1])
plt.scatter([r[0],r1[0]], [r[1],r1[1]])
plt.scatter(ref_proj[0][0], ref_proj[0][1])
plt.show()

"""