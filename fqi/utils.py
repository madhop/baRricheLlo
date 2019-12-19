action_cols = ['aSteerWheel', 'pBrakeF', 'rThrottlePedal']
prev_action_cols = ['prev'+a for a in action_cols]

# reduced
"""state_cols = ['xCarWorld', 'yCarWorld',
       'nYawBody', 'nEngine', 'positionRho', 'positionTheta',
       'referenceCurvature', 'actualCurvature',
       'actualSpeedModule', 'speedDifferenceVectorModule', 'actualAccelerationX',
       'actualAccelerationY', 'referenceAccelerationX',
       'referenceAccelerationY', 'accelerationDiffX', 'accelerationDiffY'] + prev_action_cols"""

state_cols = ['xCarWorld', 'yCarWorld',
       'nYawBody', 'nEngine', 'NGear', 'positionRho', 'positionTheta',
       'positionReferenceX', 'positionReferenceY', 'positionRelativeX',
       'positionRelativeY', 'referenceCurvature', 'actualCurvature',
       'actualSpeedModule', 'speedDifferenceVectorModule',
       'speedDifferenceOfModules', 'actualAccelerationX',
       'actualAccelerationY', 'referenceAccelerationX',
       'referenceAccelerationY', 'accelerationDiffX', 'accelerationDiffY'] + prev_action_cols



knn_state_cols = ['xCarWorld', 'yCarWorld']

ref_traj_cols = ['curLapTime', 'Acceleration_x', 'Acceleration_y', 'speed_x', 'speed_y', 'xCarWorld', 'yCarWorld', 'alpha_step']

state_prime_cols = ['prime_'+c for c in state_cols]
