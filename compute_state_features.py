import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ref_df = pd.read_csv('ref_trajectory_monza.csv') # reference trajectory
car_df = pd.read_csv('car_trajectory_monza.csv') # car trajectory

def distance(r,r1,p):   # distance line-point
    return np.linalg.norm((r1[1]-r[1])*p[0]-(r1[0]-r[0])*p[1]+r1[0]*r[1]-r1[1]*r[0])/np.linalg.norm(r1-r)

def realtive_features(p, r):
    """
        realtive feature with new reference system are needed
    """
    if r[1] > 0:
        rotation_angle = -np.arccos(np.dot(np.array([1,0]),r)/(1 * np.linalg.norm(r)))
    else:
        rotation_angle = np.arccos(np.dot(np.array([1,0]),r)/(1 * np.linalg.norm(r)))
    rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],[np.sin(rotation_angle), np.cos(rotation_angle)]])
    rel_p = np.dot(rotation_matrix,p)
    return rel_p

# Position Features
def position(r, r1, p): # inputs must be np.array
    """
        p: actual position vector at time t
        r: nn reference position vector
        r1: next reference position vector
    """
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
    rel_p2 = realtive_features(p2-p1, p-p1)
    c = np.arccos(np.dot(p-p1,p2-p1)/(np.linalg.norm(p-p1) * np.linalg.norm(p2-p1)))
    if rel_p2[1] < 0:
        c = -c
    return c

def velocity_acceleration(p, r):
    """ if you pass speed it will compute speed features,
        acceleration otherwise
        p: actual velocity or acceleration vector
        r: reference velocity or acceleration vector
    """
    actual_module = np.linalg.norm(p)
    ref_module = np.linalg.norm(r)
    diff_module = np.linalg.norm(p - r)
    diff_of_modules = ref_module - actual_module
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
"""plt.plot([0, vr[0], vp[0]],[0, vr[1], vp[1]], 'o')
plt.plot([vp[0]],[vp[1]], '*')
plt.show()"""

"""plt.arrow(r[0],r[1], (r1-r)[0], (r1-r)[1], head_width=0.01, head_length=0.1, fc='r', ec='r')
plt.arrow(r[0],r[1], (p-r)[0], (p-r)[1], head_width=0.01, head_length=0.1, fc='g', ec='g')
plt.plot([r[0], r1[0], p[0]],[r[1], r1[1], p[1]], 'o')
plt.show()"""

# find nearest neighbour in reference trajectory
def nn_ahead(p, last_ref = 0):
    print('last_ref: ', last_ref)
    j = last_ref
    found = False
    while not found:
        r = ref_df.iloc[j-1]
        r = np.array([r['x'], r['y']])
        r1 = ref_df.iloc[j]
        r1 = np.array([r1['x'], r1['y']])
        r2 = ref_df.iloc[j+1]
        r2 = np.array([r2['x'], r2['y']])
        rel = realtive_features(p-r, r1-r)
        rel1 = realtive_features(p-r1, r2-r1)
        if rel[0] > 0 and rel1[0] < 0:
            """plt.plot([r[0], r1[0], r2[0]],[r[1], r1[1], r2[1]], 'o')
            plt.plot([p[0]],[p[1]], '*')
            plt.show()"""
            print('NN is at ', j)
            found = True
            last_ref = j
        j += 1
    return j

last_ref = 0
for i in range(1,200,10):
#for i in range(car_trajectory_df.shape[0]):
    p = car_df.iloc[i]
    p_1 = car_df.iloc[i-1]
    p_2 = car_df.iloc[i-2]
    nn = nn_ahead(np.array([p['x'], p['y']]), last_ref)
    last_ref = nn
    r = ref_df.iloc[nn]
    r1 = ref_df.iloc[nn+1]
    r_1 = ref_df.iloc[nn-1]
    v_actual_module, v_ref_module, v_diff_module, v_diff_of_modules, v_angle = velocity_acceleration(np.array([p['speed_x'], p['speed_y']]), np.array([r['speed_x'], r['speed_y']]))
    tp = p['curLapTime'] - p_1['curLapTime'] # elapsed time between time t ant t-1
    tr = r['curLapTime'] - r_1['curLapTime'] # elapsed time between time t ant t-1
    ap = np.array([(p['speed_x'] - p_1['speed_x']) / tp, (p['speed_y'] - p_1['speed_y']) / tp])
    ar = np.array([(r['speed_x'] - r1['speed_x']) / tr, (r['speed_y'] - r1['speed_y']) / tr])
    a_actual_module, a_ref_module, a_diff_module, a_diff_of_modules, a_angle = velocity_acceleration(ap, ar)
    r = np.array([r['x'], r['y']])
    r1 = np.array([r1['x'], r1['y']])
    r_1 = np.array([r_1['x'], r_1['y']])
    p = np.array([p['x'], p['y']])
    rel_p, rho, theta = position(r, r1, p)
    actual_c = curvature(p, np.array([p_1['x'], p_1['y']]), np.array([p_2['x'], p_2['y']]))
    ref_c = curvature(r1, r, r_1)
