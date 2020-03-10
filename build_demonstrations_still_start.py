#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:24:38 2020

@author: umberto
"""

import sys
import os
import pickle
import argparse
import pandas as pd
import numpy as np

from fqi.et_tuning import run_tuning
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *

#%%
simulations = pd.read_csv('./trajectory/dataset_offroad.csv',
                              dtype={'isReference': bool, 'is_partial':bool})

ref_tr = pd.read_csv('./trajectory/ref_traj.csv')
ref_tr.columns = ['time', 'Acceleration_x', 'Acceleration_y', 'speed_x', 'speed_y',
       'xCarWorld', 'yCarWorld', 'alpha_step']

#%% right laps
all_laps = np.unique(simulations.NLap)
lap_times = map(lambda lap: simulations[simulations.NLap == lap]['time'].values[-1], all_laps)
ref_time = ref_tr['time'].values[-1]
perc_deltas = list(map(lambda t: (abs(t - ref_time) / ref_time * 100) <= 1.5, lap_times))
right_laps = all_laps[perc_deltas]
right_laps
#%% right laps df
right_laps_df = pd.DataFrame()
for i, lap in enumerate(right_laps):
    lap_df = simulations[simulations.NLap == lap]
    lap_df['NLap'] = i+1
    right_laps_df = right_laps_df.append(lap_df, ignore_index = True)

right_laps_df.colums = simulations.columns

#%% import still start df
still_start_df = pd.read_csv('trajectory/dataset_still_starts.csv')
#%% demonstration df
n_laps = right_laps_df.tail(1)['NLap'].values

demonstrations_df = right_laps_df.copy()
still_start_df.NLap = still_start_df.NLap + 15
#%% append still start laps to demonstrations

demonstrations_df = demonstrations_df.append(still_start_df, ignore_index = True)

#%% save demonstrations csv
demonstrations_df.to_csv(path_or_buf = "trajectory/demonstrations.csv", index = False, header = True)