#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 14:09:44 2020

@author: umberto
Run Driver Controller and plot results
"""
from driver_controller import Controller
from fqi.reward_function import *
from fqi.utils import *
from gym_torcs_ctrl import TorcsEnv

import numpy as np
import pandas as pd

from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

#%% Build env
ref_df = pd.read_csv('trajectory/ref_traj.csv')
simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 9., 14., 16., 17., 20., 47., 49., 55., 59., 60., 61., 62., 63., 65., 68.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)
env = TorcsEnv(reward_function,collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
           gear_change=False, brake=True, start_env=False, damage_th=5, slow=False, faster=False, graphic=True)

#%% play game and store data
#Tu = 3.
"""s_Kp = 0.65
s_Ki = 0.4#1.2*(Kp/0.45)/Tu #0.2    
s_Kd = 0.7#(3*(Kp/0.45)*Tu)/40"""

s_Kp = 2.2
s_Ki = 0.2#1.1
s_Kd = 0.15

t_Kp=0.1
t_Ki=0
t_Kd=0.001

b_Kp=0.3
b_Ki=0
b_Kd=0.0015

max_steps=100000
C = Controller(env, s_Kp=s_Kp,s_Ki=s_Ki,s_Kd=s_Kd, t_Kp=t_Kp,t_Ki=t_Ki,t_Kd=t_Kd, b_Kp=b_Kp,b_Ki=b_Ki,b_Kd=b_Kd)
step=0
action_vars = {'action':[],'rho':[],'speed_error':[], 'ref_steer':[],'ref_throttle':[],'ref_brake':[],
               'x':[], 'y':[], 'ref_x':[], 'ref_y':[], 
               's_integral':[], 's_derivative':[],
               't_integral':[], 't_derivative':[],
               'b_integral':[], 'b_derivative':[],}
ob = C.env.reset(relaunch=True)
for _ in range(max_steps):
    action, rho, speed_error, ref_id, x, y = C.act(ob)
    print(action)
    # applay action
    ob, reward, done, _ = C.env.step(action)
    # save data
    action_vars['action'].append(action)
    action_vars['rho'].append(rho)
    action_vars['speed_error'].append(speed_error)
    action_vars['ref_steer'].append(C.ref_df['Steer'].values[ref_id])
    action_vars['ref_throttle'].append(C.ref_df['Throttle'].values[ref_id])
    action_vars['ref_brake'].append(C.ref_df['Brake'].values[ref_id])
    action_vars['x'].append(x)
    action_vars['y'].append(y)
    action_vars['ref_x'].append(C.ref_df['xCarWorld'].values[ref_id])
    action_vars['ref_y'].append(C.ref_df['yCarWorld'].values[ref_id])
    action_vars['s_integral'].append(C.s_integral)
    action_vars['s_derivative'].append(C.s_derivative)
    action_vars['t_integral'].append(C.t_integral)
    action_vars['t_derivative'].append(C.t_derivative)
    action_vars['b_integral'].append(C.b_integral)
    action_vars['b_derivative'].append(C.b_derivative)
    
    step += 1
    if done:
        break
        
C.env.end()

#%% plot steering PID staff
fig, axs = plt.subplots(2, 1)
axs[0].set_title('Steering - Kp:'+str(s_Kp)+ ' Ki:'+str(s_Ki)+ ' Kd:'+str(s_Kd))
axs[0].plot(list(map(lambda x: x*s_Kp, action_vars['rho'])), label='rho')
axs[0].plot(list(map(lambda x: x*s_Ki, action_vars['s_integral'])), label='integral')
axs[0].plot(list(map(lambda x: x*s_Kd, action_vars['s_derivative'])), label='derivative')
axs[0].grid(True)
axs[0].legend()

axs[1].plot([x[0][0] for x in action_vars['action']], label='action')
axs[1].plot(action_vars['ref_steer'], label='ref steer')
axs[1].grid(True)
axs[1].legend()

"""m = 0
axs[2].scatter(action_vars['ref_x'][m:], action_vars['ref_y'][m:], label='ref', s=0.5)
axs[2].scatter(action_vars['x'][m:], action_vars['y'][m:], label='car', s=0.5)
axs[2].legend()"""
plt.show()

#%% plot just position
m=-300
n=-1
plt.title('Position - Steering Kp:'+str(t_Kp)+ ' Ki:'+str(t_Ki)+ ' Kd:'+str(t_Kd))
plt.scatter(action_vars['ref_x'][m:n], action_vars['ref_y'][m:n], label='ref', s=0.5)
plt.scatter(action_vars['x'][m:n], action_vars['y'][m:n], label='car', s=0.5)
plt.legend()

#%% plot throttle PID staff
fig, axs = plt.subplots(2, 1)
m = -500
axs[0].set_title('Throttle - Kp:'+str(t_Kp)+ ' Ki:'+str(t_Ki)+ ' Kd:'+str(t_Kd))
axs[0].plot(list(map(lambda x: x*t_Kp, action_vars['speed_error'][m:])), label='error')
axs[0].plot(list(map(lambda x: x*t_Ki, action_vars['t_integral'][m:])), label='integral')
axs[0].plot(list(map(lambda x: x*t_Kd, action_vars['t_derivative'][m:])), label='derivative')
axs[0].grid(True)
axs[0].legend()

axs[1].plot([x[2][0] for x in action_vars['action'][m:]], label='action')
axs[1].plot(action_vars['ref_throttle'][m:], label='ref throttle')
axs[1].grid(True)
axs[1].legend()

"""axs[2].scatter(action_vars['ref_x'], action_vars['ref_y'], label='ref', s=0.5)
axs[2].scatter(action_vars['x'], action_vars['y'], label='car', s=0.5)
axs[2].legend()"""
plt.show()

#%% just THROTTLE
plt.title('Throttle - Kp:'+str(t_Kp)+ ' Ki:'+str(t_Ki)+ ' Kd:'+str(t_Kd))
plt.plot(list(map(lambda x: x*t_Kp, action_vars['speed_error'])), label='error')
plt.plot(list(map(lambda x: x*t_Ki, action_vars['t_integral'])), label='integral')
plt.plot(list(map(lambda x: x*t_Kd, action_vars['t_derivative'])), label='derivative')
plt.grid(True)
plt.legend()


#%% plot brake PID staff
m = -600
fig, axs = plt.subplots(2, 1)
axs[0].set_title('Brake - Kp:'+str(b_Kp)+ ' Ki:'+str(b_Ki)+ ' Kd:'+str(b_Kd))
axs[0].plot(list(map(lambda x: -x*b_Kp, action_vars['speed_error'][m:])), label='error')
axs[0].plot(list(map(lambda x: -x*b_Ki, action_vars['b_integral'][m:])), label='integral')
axs[0].plot(list(map(lambda x: -x*b_Kd, action_vars['b_derivative'][m:])), label='derivative')
axs[0].grid(True)
axs[0].legend()

axs[1].plot([x[1][0] for x in action_vars['action']][m:], label='action')
axs[1].plot(action_vars['ref_brake'][m:], label='ref brake')
axs[1].grid(True)
axs[1].legend()

"""axs[2].scatter(action_vars['ref_x'], action_vars['ref_y'], label='ref', s=0.5)
axs[2].scatter(action_vars['x'], action_vars['y'], label='car', s=0.5)
axs[2].legend()"""
plt.show()

#%% plot just brake
m=-200
plt.title('Brake - Kp:'+str(b_Kp)+ ' Ki:'+str(b_Ki)+ ' Kd:'+str(b_Kd))
plt.plot([x[1][0] for x in action_vars['action'][m:]], label='action')
plt.plot(action_vars['ref_brake'][m:], label='ref brake')
plt.grid(True)
plt.legend()
#%% plot 3 actions
fig, axs = plt.subplots(3, 1)
axs[0].set_title('Steering - Kp:'+str(s_Kp)+ ' Ki:'+str(s_Ki)+ ' Kd:'+str(s_Kd))
axs[0].plot([x[0][0] for x in action_vars['action']], label='action')
axs[0].plot(action_vars['ref_steer'], label='ref steer')
axs[0].grid(True)
axs[0].legend()

axs[1].set_title('Throttle - Kp:'+str(t_Kp)+ ' Ki:'+str(t_Ki)+ ' Kd:'+str(t_Kd))
axs[1].plot([x[2][0] for x in action_vars['action']], label='action')
axs[1].plot(action_vars['ref_throttle'], label='ref throttle')
axs[1].grid(True)
axs[1].legend()

axs[2].set_title('Brake - Kp:'+str(b_Kp)+ ' Ki:'+str(b_Ki)+ ' Kd:'+str(b_Kd))
axs[2].plot([x[1][0] for x in action_vars['action']], label='action')
axs[2].plot(action_vars['ref_brake'], label='ref brake')
axs[2].grid(True)
axs[2].legend()
plt.show()