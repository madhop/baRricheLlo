#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 19:48:48 2020

@author: umberto
Driver controller
"""

import numpy as np

class Controller():
    def __init__(self):
        # Init
        self.alpha1 = None
        self.k1 = None
        self.beta1 = None
        self.k2 = None
        self.gamma1 = None
        self.gamma2 = None
        
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    def act(self, obs):
        # Find projection on reference trajectory
        
        Vref = None
        V = None
        p = None    # position of the car wrt the reference
        delta_O = None  # delta orientation of the car
        # Compute actions
        throttle = self.sigmoid(self.alpha1 * (Vref - V) + self.k1 * np.power(V, 2))
        brake = self.sigmoid(self.beta1 * (V - Vref) + self.k2 * np.power(V, 2))
        steer = np.tanh(self.gamma1 * p + self.gamma2 * delta_O)
    
    
    

if __name__ == '__main__':
    C = Controller()
    