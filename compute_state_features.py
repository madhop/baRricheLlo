import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ref_df = pd.read_csv('ref_trajectory_monza.csv') # reference trajectory
car_df = pd.read_csv('car_trajectory_monza.csv') # car trajectory

# Position Features
def position(r, r1, p): # inputs must be np.array
    rotation_angle = np.arccos(np.dot(np.array([1,0]),r1-r)/(1 * np.linalg.norm(r1-r)))
    rel_x = 0
    rel_y = 0
    rho = np.linalg.norm(r-p)
    theta = rotation_angle + np.arccos(np.dot(np.array([1,0]),p-r)/(1 * np.linalg.norm(p-r)))
    return rho, theta

def f(p):
    return p['x'], p['y']

for i in range(2):
#for i in range(car_trajectory_df.shape[0]):
    #print(i, ': ', car_trajectory_df.iloc[i][['x','y']])
    x, y = position(ref_df.iloc[0], ref_df.iloc[1] ,car_df.iloc[i])
    print(x,' ', y)
