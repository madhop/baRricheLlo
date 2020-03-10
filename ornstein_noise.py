#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:31:23 2020

@author: umberto
"""

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% initialize noise 
noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0., 0., 0.]),
                                            theta=.1, 
                                            dt=1e-2,
                                            sigma=np.array([0.05, 0.1, 0.1]),
                                            initial_noise=[0., -3., 1.5])

noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0., 0., 0.]),
                                            theta=.05, 
                                            dt=1e-2,
                                            sigma=np.array([0.05, 0.1, 0.1]),
                                            initial_noise=[0., -3., 1.5])

noise = OrnsteinUhlenbeckActionNoise(mean=np.array([0., 0., 0.]),
                                            theta=.02, 
                                            dt=1e-2,
                                            sigma=np.array([0.05, 0.05, 0.05]),
                                            initial_noise=[0., -3., 1.5])

#%% reproduce noise
noise.reset()
d = {0:[], 1:[], 2:[]}
for _ in range(200000):
    n = noise()
    d[0].append(n[0])
    d[1].append(n[1])
    d[2].append(n[2])
    
    
#%% plot noise trend
plt.plot(d[0])
plt.plot(d[1])
plt.plot(d[2])
plt.show()

#%% open logs
with open('../bc_ddpg/start_demonstrations/log_ddpgbc_0_[64, 64]_tanh_3500_20000_1_1_1.pkl', 'rb') as file:
    log_bc64 = pickle.load(file)
    
#%% plot logs
plt.plot(log_bc64['train_loss'])
plt.plot(log_bc64['val_loss'])
plt.show()