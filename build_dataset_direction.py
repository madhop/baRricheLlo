#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:29:08 2020

@author: umberto
"""

import pandas as pd
import numpy as np

#%% import dataset
simu_df = pd.read_csv('trajectory/demonstrations.csv')
columns = simu_df.columns

#%% add empty row at top
new_row = pd.DataFrame([[0]*len(columns)], columns=columns)
new_df = pd.concat([new_row, simu_df])

new_df.drop(new_df.tail(1).index, inplace = True)
#%% subtract the dataframe
a = simu_df[['xCarWorld', 'yCarWorld']].values
aa = new_df[['xCarWorld', 'yCarWorld']].values
direction = a-aa
#%% direction
simu_df['direction_x'] = direction[:,0]
simu_df['direction_y'] = direction[:,1]



#%% add empty row at top and compute lap beginnings
new_row = pd.DataFrame([[0]*len(simu_df.columns)], columns=simu_df.columns)
new_df = pd.concat([new_row, simu_df])

new_df.drop(new_df.tail(1).index, inplace = True)
new_df.reset_index(inplace = True)


lap_beginnings = (simu_df.time - new_df.time ) < 0








#%% drop wrong rows
simu_df.drop(lap_beginnings[lap_beginnings == True].index, inplace = True)
simu_df.drop(0, inplace = True)



#%% export to csv
simu_df.to_csv('trajectory/demonstrations.csv')