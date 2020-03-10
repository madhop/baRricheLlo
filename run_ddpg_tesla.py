#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:42:54 2020

@author: umberto

"""
from joblib import Parallel, delayed
import os
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.direction_feature import create_direction_feature
from fqi.utils import action_cols, ref_traj_cols, penalty_cols, state_dict, state_cols
import pandas as pd
import numpy as np
import argparse
import itertools
import pickle

from stable_baselines.gail.dataset.dataset import ExpertDataset
from stable_baselines.ddpg.policies import MlpPolicy
from ddpg.ddpg_driver import DDPG
from gym_torcs import TorcsEnv

import tensorflow as tf

def run_training_online(demo_name, out_dir):
    print('state_cols:', state_cols)
    
    # load reference trajectory
    ref_df = pd.read_csv('trajectory/ref_traj.csv')
    ref_df.columns = ref_traj_cols
    simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
    simulations = create_direction_feature(simulations)
    # Reward function
    penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
    right_laps = np.array([ 9., 14., 16., 17., 20., 47., 49., 55., 59., 60., 61., 62., 63., 65., 68.])
    penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
    reward_function = Temporal_projection(ref_df, penalty=penalty)
    env = TorcsEnv(reward_function,collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=0, slow=False, faster=False, graphic=False)
    
    # load model
    model = DDPG.load('./model_file/'+demo_name+'/ddpgbc_0_[64, 64]_tanh_3500_20000_1_1_1.zip')
    model.env = env
    
    # run training and save model
    for _ in range(50000):
        model.learn(total_timesteps=25)
        name = 'ddpgbc_{}_model'.format(demo_name)
        model.save(out_dir + name)
        model.save('model_file/' + output_model)
        print('save model')


if __name__ == '__main__':
    
    print('Started experiments')
    run_training_online('demonstrations', '../ddpg_bc/')
    print('Experiments terminated')