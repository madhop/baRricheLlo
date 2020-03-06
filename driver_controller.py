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
    
    
    

if __name__ == '__main__':
    C = Controller()
    