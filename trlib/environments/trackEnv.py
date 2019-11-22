import gym
import numpy as np
from gym import spaces
import math

"""
The Puddleworld environment

Info
----
  - State space: 2D Box (x,y)
  - Action space: Discrete (UP,DOWN,RIGHT,LEFT)
  - Parameters: goal position x and y, puddle centers, puddle variances

References
----------
  
  - Andrea Tirinzoni, Andrea Sessa, Matteo Pirotta, Marcello Restelli.
    Importance Weighted Transfer of Samples in Reinforcement Learning.
    International Conference on Machine Learning. 2018.
    
  - https://github.com/amarack/python-rl/blob/master/pyrl/environments/puddleworld.py
"""

class TrackEnv(gym.Env):

    def __init__(self, state_dim, action_dim, gamma, action_space):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.action_space = action_space