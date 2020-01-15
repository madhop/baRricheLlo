import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

fqi_exp_path = os.path.join('..', 'fqi_experiments')
data_path = os.path.join('..', '..', '..', '..', '..', '..', '..', 'data', 'ferrari', 'driver', 'datasets',
                         'csv')

action_cols = ['pBrakeF', 'aSteerWheel', 'rThrottlePedal']#, 'BDownshiftRequested', 'BUpshiftRequested']

prev_action_cols = ['prev' + a for a in action_cols]

ref_curvature_features = ['referenceCurvature' + str(i) for i in range(100)]

#state_cols = ['xCarWorld', 'yCarWorld', 'positionReferenceX', 'positionReferenceY',
#              'nYawBody', 'nEngine', 'NGear', 'positionRho',
#              'positionTheta', 'positionLeft', 'positionRight',
#              'positionRelativeX', 'positionRelativeY', 'actualCurvature', 'referenceCurvature',
#              'actualSpeedModule', 'referenceSpeedAngle', 'speedDifferenceVectorModule',
#              'speedDifferenceOfModules', 'actualAccelerationModule', 'referenceAccelerationAngle',
#              'accelerationDifferenceVectorModule', 'accelerationDifferenceOfModules',
#              'actualAccelerationX', 'actualAccelerationY',
#              'accelerationDiffX', 'accelerationDiffY'] + prev_action_cols


state_cols = ['xCarWorld', 'yCarWorld', 'positionRho', 'positionTheta',
              'actualCurvature', 'referenceCurvature',
              'actualSpeedModule', 'speedDifferenceOfModules',
              'actualAccelerationX', 'actualAccelerationY', 'accelerationDiffX', 'accelerationDiffY',
              'nYawBody', 'nEngine', 'NGear'] + prev_action_cols

state_prime_cols = [s + '_prime' for s in state_cols]

knn_state_cols = ['xCarWorld', 'yCarWorld']

episode_col = ['NLap']

car_cols = ['ABal_e_Body_YRS_Sim_HS', 'ABal_e_Body_YRS_Sim_LS', 'ABal_e_Tot_YRS_Sim_HS', 'ABal_e_Tot_YRS_Sim_LS',
            'AeroBal_Setup_Sim_HS', 'AeroBal_Setup_Sim_LS', 'CLt_e_Body_YRS_Sim_HS', 'CLt_e_Body_YRS_Sim_LS',
            'CLt_e_Tot_YRS_Sim_HS', 'CLt_e_Tot_YRS_Sim_LS', 'CarBal_Setup_Sim_HS', 'CarBal_Setup_Sim_LS',
            'CarBal_Setup_YRS_Sim_HS', 'CarBal_Setup_YRS_Sim_LS', 'RideFr_Setup_Sim_HS', 'RideFr_Setup_Sim_LS',
            'RideRe_Setup_Sim_HS', 'RideRe_Setup_Sim_LS', 'RollBal_Setup_Sim_HS', 'RollBal_Setup_Sim_LS',
            'Roll_stf_sim_LS']


def rotate_and_translate(px, py, alpha, ox, oy):
    """Given the coordinates of a points it computes the new coordinates
    applying rotation of alpha and translation of ox,oy
    
    Input:
    px,py -- x,y coordinates of the point
    alpha -- rotation angle
    ox,oy -- x,y coordinates of the reference system
    
    Ouput:
    (x_new, y_new) -- x,y new coordinates
    
    """
    rotate_x = lambda x, y: x * np.cos(alpha) + y * np.sin(alpha)
    rotate_y = lambda x, y: (-x * np.sin(alpha) + y * np.cos(alpha))

    # Rotation of the point
    x_rot = rotate_x(px, py)
    y_rot = rotate_y(px, py)

    # Rotation of the new center
    ox_rot = rotate_x(ox, oy)
    oy_rot = rotate_y(ox, oy)

    # Translation
    x_new = x_rot - ox_rot
    y_new = y_rot - oy_rot

    return x_new, y_new

def vectorized_rotate_and_translate(p, alpha, o):
    x_rotation = np.column_stack((np.cos(alpha), np.sin(alpha)))
    y_rotation = np.column_stack((-np.sin(alpha), np.cos(alpha)))
    rotated_px = np.sum(x_rotation * p, axis=1)
    rotated_py = np.sum(y_rotation * p, axis=1)
    rotated_ox = np.sum(x_rotation * o, axis=1)
    rotated_oy = np.sum(y_rotation * o, axis=1)
    rotated_p = np.column_stack((rotated_px, rotated_py))
    rotated_o = np.column_stack((rotated_ox, rotated_oy))
    return rotated_p - rotated_o



def create_state_thresholds(simulation, k, n_jobs):
    scaler = MinMaxScaler()
    action_values = scaler.fit_transform(simulation[action_cols].values)
    kmeans = KMeans(n_clusters=k, n_jobs=n_jobs).fit(action_values)

    states = simulation[state_cols].values
    max_variation = np.zeros((len(state_cols), k))
    for i in range(k):
        mask = kmeans.labels_ == i
        mv = states[mask].max(axis=0) - states[mask].min(axis=0)
        max_variation[:, i] = np.array(mv)

    thresholds = np.mean(max_variation, axis=1)
    return thresholds
