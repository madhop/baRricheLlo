import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ref_df = pd.read_csv('ref_trajectory_monza.csv') # reference trajectory
car_df = pd.read_csv('car_trajectory_monza.csv') # car trajectory

def distance(r,r1,p):   # distance line-point
    return np.linalg.norm((r1[1]-r[1])*p[0]-(r1[0]-r[0])*p[1]+r1[0]*r[1]-r1[1]*r[0])/np.linalg.norm(r1-r)

def realtive_features(p, r):
    if r[1] > 0:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),r)/(1 * np.linalg.norm(r)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),r)/(1 * np.linalg.norm(r)))
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    rel_p = np.dot(rotation_matrix,p)
    return rel_p

# Position Features
def position(r, r1, p): # inputs must be np.array
    """if r1[1] > r[1]:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),r1-r)/(1 * np.linalg.norm(r1-r)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),r1-r)/(1 * np.linalg.norm(r1-r)))
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    rel_p = np.dot(rotation_matrix,p-r)"""
    rel_p = realtive_features(p-r, r1-r)
    rho = np.linalg.norm(p-r)
    if np.array_equal(p-r, np.array([0,0])):
        theta = 0
    elif rel_p[1] < 0:  # p is on the right of r
        theta = -np.arccos(np.dot(np.array([1,0]),rel_p)/(1 * np.linalg.norm(rel_p)))
    else:
        theta = np.arccos(np.dot(np.array([1,0]),rel_p)/(1 * np.linalg.norm(rel_p)))
    return rel_p, rho, theta

def curvature(p, p1, p2):   # p_t, p_t-1 and p_t-2
    """if p[1] > p1[1]:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),p-p1)/(1 * np.linalg.norm(p-p1)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),p-p1)/(1 * np.linalg.norm(p-p1)))
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    rel_p2 = np.dot(rotation_matrix,p2-p1)"""
    rel_p2 = realtive_features(p2-p1, p-p1)
    c = np.arccos(np.dot(p-p1,p2-p1)/(np.linalg.norm(p-p1) * np.linalg.norm(p2-p1)))
    if rel_p2[1] < 0:
        c = -c
    return c

def velocity_acceleration(p, r):
    """ if you pass speed it will compute speed features,
        acceleration otherwise
    """
    actual_module = np.linalg.norm(p)
    ref_module = np.linalg.norm(r)
    diff_module = np.linalg.norm(p - r)
    diff_of_modules = ref_module - actual_module

    """if r[1] > 0:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),r)/(1 * np.linalg.norm(r)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),r)/(1 * np.linalg.norm(r)))
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    rel_p = np.dot(rotation_matrix,p)"""
    rel_p = realtive_features(p, r)
    angle = np.arccos( np.dot(r,p) / (np.linalg.norm(p) * np.linalg.norm(r)))
    if rel_p[1] < 0:
        angle = -angle
    return actual_module, ref_module, diff_module, diff_of_modules, angle

"""# test state features
r = np.array([1,1])
r1 = np.array([0,1])
p = np.array([1,2])"""

r = np.array([ref_df.iloc[0]['x'], ref_df.iloc[0]['y']])
r1 = np.array([ref_df.iloc[10]['x'], ref_df.iloc[10]['y']])
p = np.array([car_df.iloc[0]['x'], car_df.iloc[0]['y']])
rel_p, rho, theta = position(r, r1, p)
cur = curvature(r1, r, p)

"""vp = np.array([car_df.iloc[0]['speed_x'], car_df.iloc[0]['speed_y']])
vr = np.array([ref_df.iloc[0]['speed_x'], ref_df.iloc[0]['speed_y']])"""
vr = np.array([1,1])
vp = np.array([0.5,0.5])
r = ref_df.iloc[10]
r1 = ref_df.iloc[9]
p = car_df.iloc[10]
p1 = car_df.iloc[9]
tp = p['curLapTime'] - p1['curLapTime'] # elapsed time between time t ant t-1
tr = r['curLapTime'] - r1['curLapTime'] # elapsed time between time t ant t-1
ap = np.array([(p['speed_x'] - p1['speed_x']) / tp, (p['speed_y'] - p1['speed_y']) / tp])
ar = np.array([(r['speed_x'] - r1['speed_x']) / tr, (r['speed_y'] - r1['speed_y']) / tr])
velocity_acceleration(vp, vr)
velocity_acceleration(ap, ar)


#plt.arrow(0,0, vr[0], vr[1], head_width=0.01, head_length=0.1, fc='r', ec='r')
#plt.arrow(0,0, vp[0], vp[1], head_width=0.01, head_length=0.1, fc='g', ec='g')
plt.plot([0, vr[0], vp[0]],[0, vr[1], vp[1]], 'o')
plt.plot([vp[0]],[vp[1]], '*')
plt.show()

"""plt.arrow(r[0],r[1], (r1-r)[0], (r1-r)[1], head_width=0.01, head_length=0.1, fc='r', ec='r')
plt.arrow(r[0],r[1], (p-r)[0], (p-r)[1], head_width=0.01, head_length=0.1, fc='g', ec='g')
plt.plot([r[0], r1[0], p[0]],[r[1], r1[1], p[1]], 'o')
plt.show()"""

"""for i in range(2):
#for i in range(car_trajectory_df.shape[0]):
    r = np.array([ref_df.iloc[0]['x'], ref_df.iloc[0]['y']])
    r1 = np.array([ref_df.iloc[10]['x'], ref_df.iloc[10]['y']])
    p = np.array([car_df.iloc[0]['x'], car_df.iloc[0]['y']])
    rel_p, rho, theta = position(r, r1, p)
    print(rho,' ', theta)
"""
