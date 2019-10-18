import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ref_df = pd.read_csv('ref_trajectory_monza.csv') # reference trajectory
car_df = pd.read_csv('car_trajectory_monza.csv') # car trajectory

def distance(r,r1,p):   # distance line-point
    return np.linalg.norm((r1[1]-r[1])*p[0]-(r1[0]-r[0])*p[1]+r1[0]*r[1]-r1[1]*r[0])/np.linalg.norm(r1-r)

# Position Features
def position(r, r1, p): # inputs must be np.array
    if r1[1] > r[1]:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),r1-r)/(1 * np.linalg.norm(r1-r)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),r1-r)/(1 * np.linalg.norm(r1-r)))

    print("rotation_angle: ", rotation_angle)
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    print("rotation_matrix: \n", rotation_matrix)
    rel_p = np.dot(rotation_matrix,p-r)#np.array([(p-r)[0]*np.cos(rotation_angle)-(p-r)[1]*np.sin(rotation_angle), (p-r)[0]*np.sin(rotation_angle)+(p-r)[1]*np.cos(rotation_angle)])
    print("rel_p: ", rel_p)
    #print("dist: ", distance(r,r1,p))
    rho = np.linalg.norm(p-r)
    if np.array_equal(p-r, np.array([0,0])):
        theta = 0
    elif rel_p[1] < 0:  # p is on the right of r
        theta = -np.arccos(np.dot(np.array([1,0]),rel_p)/(1 * np.linalg.norm(rel_p)))
    else:
        theta = np.arccos(np.dot(np.array([1,0]),rel_p)/(1 * np.linalg.norm(rel_p)))
    print('theta: ', theta)
    return rel_p, rho, theta

"""# test state features
r = np.array([1,1])
r1 = np.array([2,2])
p = np.array([1,0])"""

r = np.array([ref_df.iloc[0]['x'], ref_df.iloc[0]['y']])
r1 = np.array([ref_df.iloc[10]['x'], ref_df.iloc[10]['y']])
p = np.array([car_df.iloc[0]['x'], car_df.iloc[0]['y']])
rel_p, rho, theta = position(r, r1, p)


plt.arrow(r[0],r[1], (r1-r)[0], (r1-r)[1], head_width=0.07, head_length=0.5, fc='k', ec='k')
plt.plot([r[0], r1[0], p[0]],[r[1], r1[1], p[1]], '*')
plt.show()

"""for i in range(2):
#for i in range(car_trajectory_df.shape[0]):
    #print(i, ': ', car_trajectory_df.iloc[i][['x','y']])
    r = np.array([ref_df.iloc[0]['x'], ref_df.iloc[0]['y']])
    r1 = np.array([ref_df.iloc[10]['x'], ref_df.iloc[10]['y']])
    p = np.array([car_df.iloc[0]['x'], car_df.iloc[0]['y']])
    rho, theta = position(r, r1, p)
    print(rho,' ', theta)
"""
