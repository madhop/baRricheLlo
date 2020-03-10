#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:52:50 2020

@author: umberto
"""

from bc_tuning import *

"""taskset -c 32-47 python3 bc_tuning.py --state_type 0 --batch_size 3500 --epochs 10000 --layers 64 64 --activation tanh --n_jobs -1 --action_weights 1.0 0.5 0.5 --out_dir ../bc_normalization/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 --batch_size 3500 --epochs 10000 --layers 64 64 --activation tanh --n_jobs -1 --out_dir ../bc_normalization/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 --batch_size 3500 --epochs 10000 --layers 64 64 --activation tanh --n_jobs -1 --out_dir ../bc_normalization/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 200 2000 3500 5000 --epochs 10000 --layers 64 64 --layers 32 32 --layers 128 --activation tanh relu --n_jobs -1 --out_dir ../bc_ddpg/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 200, 2000 3500 5000 --epochs 10000 --layers 64 64 --layers 32 32 --layers 128 --activation tanh relu --n_jobs -1 --out_dir ../bc_ddpg/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 2000 3500 5000 --epochs 8000 --layers 64 64 --layers 32 32 --layers 128 --activation tanh --n_jobs -1 --out_dir ../bc_ddpg/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 20 30 40 --epochs 10 --layers 64 64 --layers 32 32 --layers 128 --activation tanh --n_jobs -1 --out_dir ../bc_ddpg/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 20 30 40 --epochs 10 --layers 64 64 --layers 32 32 --layers 128 --activation tanh --learning_rate 1e-4 --n_jobs -1 --out_dir ../bc_ddpg/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 2000 3500 5000 --epochs 6000 --layers 64 64 --layers 32 32 --layers 128 --activation tanh --learning_rate 1e-4 --n_jobs -1 --out_dir ../bc_ddpg/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 20 30 40 --epochs 10 --layers 64 64 --layers 32 32 --layers 128 --activation tanh --learning_rate 1e-4 --n_jobs -1 --out_dir ../bc_ddpg/
taskset -c 32-47 python3 bc_tuning.py --state_type 0 1 --batch_size 2000 3500 5000 --epochs 6000 --layers 64 64 --layers 32 32 --layers 128 --activation tanh --learning_rate 1e-4 --n_jobs -1 --out_dir ../bc_ddpg/
python bc_tuning.py --state_type 0 1 --batch_size 2000 3500 5000 --epochs 6000 --layers 64 64 --layers 32 32 --layers 128 --activation tanh --learning_rate 0.0001 --n_jobs -1 --out_dir ../bc_ddpg/
python bc_tuning.py --state_type 0 --batch_size 3500 --epochs 20000 --layers 64 64 --activation tanh --n_jobs -1 --action_weights 1.0 0.5 0.5 --action_weights 1.0 1.0 1.0 --action_weights 1.0 0.1 0.1 --out_dir ../training_200302/start_demonstrations/ --demo_name start_demonstrations
"""

#%% params
"""state_id = 0#[0, 1]
#layers = [64, 64]
#layers = [32, 32]
#layers= 128
policy_layers= [150, 300]
activation = "tanh"
batch_size = 2000#[2000, 3500 ,5000]
epochs = 6000
action_weights = None
#learning_rate = 1e-4
#n_jobs = -1
out_dir = "../bc_ddpg/"

#policy_activation = [tf.nn.tanh if a == 'tanh' else tf.nn.relu for a in activation]
policy_activation = tf.nn.tanh if activation == 'tanh' else tf.nn.relu
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
params = (state_id, policy_layers, policy_activation, batch_size, epochs, action_weights)
#%% run experiment
print('Started experiments')
run_experiment(*params, out_dir)
print('Experiments terminated')"""

#%% many cases
state_type = [0]
#layers = [64, 64]
#layers = [32, 32]
layers= [[64, 64]]
activation = ["tanh"]
batch_size = [3500]
epochs = [20000]
action_weights = [None]#[[1.0, 1.0, 1.0]]
#learning_rate = 1e-4
out_dir = "../bc_ddpg/start_demonstrations/"
demo_name =  "start_demonstrations"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
activations = [tf.nn.tanh if a == 'tanh' else tf.nn.relu for a in activation]

params = [state_type, layers, activations, batch_size, epochs, action_weights]#, args.learning_rate]
params_comb = list(itertools.product(*params))

#%%
for i, params in enumerate(params_comb):
    print(i, params)
    print('Started experiment')
    run_experiment(*params, demo_name, out_dir)
    print('Experiment terminated')
    
print('All experiments terminated')