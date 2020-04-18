#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%% reward function
from fqi.reward_function import *
import pandas as pd

ref_df = pd.read_csv('trajectory/ref_traj.csv')
reward_function = Spatial_projection(ref_df, penalty=None)
"""
[ 685.34301758 1170.90002441]65
[ 692.75500488 1170.57995605]74
[ 700.21002197, 1170.34997559]82
"""
past_ref_id = 74
eucl_dists = np.linalg.norm(reward_function.ref_p[past_ref_id:past_ref_id+20]-np.array([ 700.21002197, 1170.34997559]), axis=1)
print(past_ref_id + np.argmin(eucl_dists))
#%% plot
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('POIS_logs/logs/eval_theta_episodes-1587046561.csv')
theta_eval = df.values
fig, axs = plt.subplots(2, 1)

for i in range(0,14,2):
    axs[0].plot(theta_eval[:,i])
    
axs[1].plot(theta_eval[:,14])


fig, axs = plt.subplots(1, 1)
axs.plot(theta_eval[:,14])