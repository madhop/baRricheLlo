action_cols = ['aSteerWheel', 'pBrakeF', 'rThrottlePedal']
prev_action_cols = ['prev'+a for a in action_cols]

# reduced
"""state_cols = ['xCarWorld', 'yCarWorld', 'positionRho', 'positionTheta',
              'actualCurvature', 'referenceCurvature',
              'actualSpeedModule', 'speedDifferenceOfModules',
              'actualAccelerationX', 'actualAccelerationY', 'accelerationDiffX', 'accelerationDiffY',
              'nYawBody', 'nEngine', 'NGear'] + prev_action_cols"""

state_cols = ['xCarWorld', 'yCarWorld',
       'nYawBody', 'nEngine', 'positionRho', 'positionTheta',
       'referenceCurvature', 'actualCurvature',
       'actualSpeedModule', 'speedDifferenceVectorModule', 'actualAccelerationX',
       'actualAccelerationY', 'referenceAccelerationX',
       'referenceAccelerationY', 'accelerationDiffX', 'accelerationDiffY'] + prev_action_cols

"""state_cols = ['xCarWorld', 'yCarWorld',
       'nYawBody', 'nEngine', 'NGear', 'positionRho', 'positionTheta',
       'positionReferenceX', 'positionReferenceY', 'positionRelativeX',
       'positionRelativeY', 'referenceCurvature', 'actualCurvature',
       'actualSpeedModule', 'speedDifferenceVectorModule',
       'speedDifferenceOfModules', 'actualAccelerationX',
       'actualAccelerationY', 'referenceAccelerationX',
       'referenceAccelerationY', 'accelerationDiffX', 'accelerationDiffY'] + prev_action_cols"""

penalty_cols = ['xCarWorld', 'yCarWorld']
#penalty_cols = state_cols



knn_state_cols = ['xCarWorld', 'yCarWorld']

ref_traj_cols = ['curLapTime', 'Acceleration_x', 'Acceleration_y', 'speed_x', 'speed_y', 'xCarWorld', 'yCarWorld', 'alpha_step']

state_prime_cols = ['prime_'+c for c in state_cols]

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
