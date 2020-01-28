from gym_torcs import TorcsEnv

import time
import os
import sys
import tensorflow as tf
import numpy as np


from utils_torcs import *
from preprocess_raw_torcs_algo import *
from build_dataset_offroad import *
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from ddpg.ddpg_driver import DDPG

model = DDPG.load("model_file/ddpg_torcs")

model.learn(total_timesteps=2000)
