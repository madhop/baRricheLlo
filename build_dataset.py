import compute_state_features as sf
import pandas as pd
import numpy as np

ref_df = pd.read_csv('trajectory/test_ref_traj.csv') # reference trajectory
car_df = pd.read_csv('raw_torcs_data/preprocessed_torcs.csv') # car trajectory         car_trajectory_monza
actual_df = pd.DataFrame()

## Compute state's features
last_ref = 0
last_lap = 0
for index, row in car_df.iterrows():
    if index <= car_df.index[2] or index >= car_df.index[-2]:   ## first and last rows
        pass
    else:
        p = row
        p_1 = car_df.iloc[index-1]
        p_2 = car_df.iloc[index-2]
        nn = sf.nn_ahead(np.array([p['x'], p['y']]), ref_df, last_ref)
        print(nn)
        if p['NLap'] != last_lap:
            last_ref = 0
            last_lap = p['NLap']
        else:
            last_ref = nn
        r = ref_df.iloc[nn]
        r1 = ref_df.iloc[nn+1]
        r_1 = ref_df.iloc[nn-1]
        v_actual_module, v_ref_module, v_diff_module, v_diff_of_modules, v_angle = sf.velocity_acceleration(np.array([p['speed_x'], p['speed_y']]), np.array([r['speed_x'], r['speed_y']]))
        #tp = p['curLapTime'] - p_1['curLapTime'] # elapsed time between time t ant t-1
        #tr = r['curLapTime'] - r_1['curLapTime'] # elapsed time between time t ant t-1
        #ap = np.array([(p['speed_x'] - p_1['speed_x']) / tp, (p['speed_y'] - p_1['speed_y']) / tp])
        #ar = np.array([(r['speed_x'] - r1['speed_x']) / tr, (r['speed_y'] - r1['speed_y']) / tr])
        ap = np.array([p['Acceleration_x'], p['Acceleration_y']])
        ar = np.array([r['Acceleration_x'], r['Acceleration_y']])
        a_actual_module, a_ref_module, a_diff_module, a_diff_of_modules, a_angle = sf.velocity_acceleration(ap, ar)
        r = np.array([r['x'], r['y']])
        r1 = np.array([r1['x'], r1['y']])
        r_1 = np.array([r_1['x'], r_1['y']])
        p = np.array([p['x'], p['y']])
        rel_p, rho, theta = position(r, r1, p)
        actual_c = curvature(p, np.array([p_1['x'], p_1['y']]), np.array([p_2['x'], p_2['y']]))
        ref_c = curvature(r1, r, r_1)


        actual_df.loc[index, 'NLap'] = p['NLap']
        actual_df.loc[index, 'time'] = p['curLapTime']
        actual_df.loc[index, 'isReference'] = p['isReference']
        actual_df.loc[index, 'is_partial'] = p['is_partial']
        actual_df.loc[index, 'xCarWorld'] = p['x']
        actual_df.loc[index, 'yCarWorld'] = p['y']
        actual_df.loc[index, 'nYawBody'] = p['yaw']
        actual_df.loc[index, 'nEngine'] = p['rpm']
        actual_df.loc[index, 'NGear'] = p['Gear']
        actual_df.loc[index, 'prevaSteerWheel'] = p_1['Steer']
        actual_df.loc[index, 'prevpBrakeF'] = p_1['Brake']
        actual_df.loc[index, 'prevrThrottlePedal'] = p_1['Throttle']
        actual_df.loc[index, 'positionRho'] = rho
        actual_df.loc[index, 'positionTheta'] = theta
        actual_df.loc[index, 'positionReferenceX'] = r['x']
        actual_df.loc[index, 'positionReferenceY'] = r['y']
        if rel_p[1] > 0:
            actual_df.loc[index, 'positionLeft'] = 0
            actual_df.loc[index, 'positionRight'] = 1
        else:
            actual_df.loc[index, 'positionLeft'] = 1
            actual_df.loc[index, 'positionRight'] = 0
        actual_df.loc[index, 'positionRelativeX'] = rel_p[0]
        actual_df.loc[index, 'positionRelativeY'] = rel_p[1]
        actual_df.loc[index, 'referenceCurvature'] = ref_c
        actual_df.loc[index, 'actualCurvature'] = actual_c
        actual_df.loc[index, 'actualSpeedModule'] = v_actual_module
        actual_df.loc[index, 'referenceSpeedAngle'] = v_angle
        actual_df.loc[index, 'speedDifferenceVectorModule'] = v_diff_module
        actual_df.loc[index, 'speedDifferenceOfModules'] = v_diff_of_modules
        actual_df.loc[index, 'actualAccelerationModule'] = a_actual_module
        actual_df.loc[index, 'referenceAccelerationAngle'] = a_angle
        actual_df.loc[index, 'accelerationDifferenceVectorModule'] = a_diff_module
        actual_df.loc[index, 'accelerationDifferenceOfModules'] = a_diff_of_modules
        actual_df.loc[index, 'aSteerWheel'] =  p['Steer']
        actual_df.loc[index, 'pBrakeF'] =  p['Break']
        actual_df.loc[index, 'rThrottlePedal'] =  p['Throttle']

actual_df.to_csv(path_or_buf = "trajectory/dataset.csv", index = False)


"""
prevBDownshiftRequested
prevBUpshiftRequested




BDownshiftRequested
BUpshiftRequested
"""
