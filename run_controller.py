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
gamma1=0.002    #rho
gamma2=(2*np.pi) * 0.1     #delta_O
gamma3=(2*np.pi) * 24      #delta_ref_O
Tu = 3.
Kp = 0.05
Ki = 0.001#1.2*(Kp/0.45)/Tu #0.2    
Kd = 0.5#(3*(Kp/0.45)*Tu)/40
alpha1=1
k1=0.000001
k2=0
max_steps=100000
C = Controller(env, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, alpha1=alpha1, k1=k1, k2=k2, s_Kp=Kp,s_Ki=Ki,s_Kd=Kd)
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
    action_vars['integral'].append(C.s_integral)
    action_vars['derivative'].append(C.s_derivative)
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

axs[2].scatter(action_vars['ref_x'], action_vars['ref_y'], label='ref', s=0.5)
axs[2].scatter(action_vars['x'], action_vars['y'], label='car', s=0.5)
axs[2].legend()
plt.show()
