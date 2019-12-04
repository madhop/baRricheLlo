action_cols = ['aSteerWheel', 'pBrakeF', 'rThrottlePedal']
prev_action_cols = ['prev'+a for a in action_cols]

state_cols = ['xCarWorld', 'yCarWorld',
       'nYawBody', 'nEngine', 'NGear', 'positionRho', 'positionTheta',
       'positionReferenceX', 'positionReferenceY', 'positionRelativeX',
       'positionRelativeY', 'referenceCurvature', 'actualCurvature',
       'actualSpeedModule', 'speedDifferenceVectorModule',
       'speedDifferenceOfModules', 'actualAccelerationX',
       'actualAccelerationY', 'referenceAccelerationX',
       'referenceAccelerationY', 'accelerationDiffX', 'accelerationDiffY'] + prev_action_cols


knn_state_cols = ['xCarWorld', 'yCarWorld']

state_prime_cols = ['prime_'+c for c in state_cols]

def compute_reward():
    reward = 0
    return reward
