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

    #print("rotation_angle: ", rotation_angle)
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    #print("rotation_matrix: \n", rotation_matrix)
    rel_p = np.dot(rotation_matrix,p-r)#np.array([(p-r)[0]*np.cos(rotation_angle)-(p-r)[1]*np.sin(rotation_angle), (p-r)[0]*np.sin(rotation_angle)+(p-r)[1]*np.cos(rotation_angle)])
    #print("rel_p: ", rel_p)
    #print("dist: ", distance(r,r1,p))
    rho = np.linalg.norm(p-r)
    if np.array_equal(p-r, np.array([0,0])):
        theta = 0
    elif rel_p[1] < 0:  # p is on the right of r
        theta = -np.arccos(np.dot(np.array([1,0]),rel_p)/(1 * np.linalg.norm(rel_p)))
    else:
        theta = np.arccos(np.dot(np.array([1,0]),rel_p)/(1 * np.linalg.norm(rel_p)))
    return rel_p, rho, theta

def curvature(p, p1, p2):   # p_t, p_t-1 and p_t-2
    """ return the curvature.
        Negative if p2 if on the right of p
    """
    if p[1] > p1[1]:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),p-p1)/(1 * np.linalg.norm(p-p1)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),p-p1)/(1 * np.linalg.norm(p-p1)))
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    rel_p2 = np.dot(rotation_matrix,p2-p1)
    c = np.arccos(np.dot(p-p1,p2-p1)/(np.linalg.norm(p-p1) * np.linalg.norm(p2-p1)))
    if rel_p2[1] < 0:
        c = -c
    return c

def velocity(vp, vr):
    actual_module = np.linalg.norm(vp)
    ref_module = np.linalg.norm(vr)
    diff_module = np.linalg.norm(vp - vr)
    diff_of_modules = ref_module - actual_module

    if vr[1] > 0:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),vr)/(1 * np.linalg.norm(vr)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),vr)/(1 * np.linalg.norm(vr)))
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    rel_vp = np.dot(rotation_matrix,vp)
    speed_angle = np.arccos( np.dot(vr,vp) / (np.linalg.norm(vp) * np.linalg.norm(vr)))
    if rel_vp[1] < 0:
        speed_angle = -speed_angle
    print('speed_angle: ', speed_angle)
    return actual_module

def acceleration():
    return 1

"""# test state features
r = np.array([1,1])
r1 = np.array([0,1])
p = np.array([1,2])"""

r = np.array([ref_df.iloc[0]['x'], ref_df.iloc[0]['y']])
r1 = np.array([ref_df.iloc[10]['x'], ref_df.iloc[10]['y']])
p = np.array([car_df.iloc[0]['x'], car_df.iloc[0]['y']])
rel_p, rho, theta = position(r, r1, p)
cur = curvature(r1, r, p)

"""vp = np.array([car_df.iloc[0]['speed_x'], car_df.iloc[10]['speed_y']])
vr = np.array([ref_df.iloc[0]['speed_x'], ref_df.iloc[10]['speed_y']])"""
vr = np.array([1,1])
vp = np.array([0.5,0.5])
velocity(vp, vr)

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
