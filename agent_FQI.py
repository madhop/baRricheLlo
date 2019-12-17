import numpy as np
import compute_state_features as csf
import pandas as pd
from scipy import spatial
import pickle
from fqi.utils import *

class AgentFQI(object):
    def __init__(self, ref_df, policy_path, action_dispatcher_path):
        self.ref_df = ref_df
        self.tree = spatial.KDTree(list(zip(ref_df['xCarWorld'], ref_df['yCarWorld'])))
        self.end_of_lap = False

        # load policy object
        with open(policy_path, 'rb') as pol:
            self.policy = pickle.load(pol)

        # load action dispatcher object
        with open(action_dispatcher_path, 'rb') as ad:
            self.action_dispatcher = pickle.load(ad)


    def make_observaton(self, ob, p_1, p_2, prev_action):
        p = ob
        nn = csf.nn_kdtree(np.array([p['x'], p['y']]), self.tree)
        # check if you are at the end of the lap
        if nn >= self.ref_df.shape[0]-1:
            nn = self.ref_df.shape[0]-2
            self.end_of_lap = True
        r = self.ref_df.iloc[nn]
        r1 = self.ref_df.iloc[nn+1]
        r_1 = self.ref_df.iloc[nn-1]

        v_actual_module, v_ref_module, v_diff_module, v_diff_of_modules, v_angle = csf.velocity_acceleration(np.array([p['speed_x'], p['speed_y']]), np.array([r['speed_x'], r['speed_y']]))
        ap = np.array([p['Acceleration_x'], p['Acceleration_y']])
        ar = np.array([r['Acceleration_x'], r['Acceleration_y']])
        a_actual_module, a_ref_module, a_diff_module, a_diff_of_modules, a_angle = csf.velocity_acceleration(ap, ar)
        rel_p, rho, theta = csf.position(np.array([r['xCarWorld'], r['yCarWorld']]), np.array([r1['xCarWorld'], r1['yCarWorld']]), np.array([p['x'], p['y']]))
        actual_c = csf.curvature(np.array([p['x'], p['y']]), np.array([p_1['x'], p_1['y']]), np.array([p_2['x'], p_2['y']]))
        ref_c = csf.curvature(np.array([r1['xCarWorld'], r1['yCarWorld']]), np.array([r['xCarWorld'], r['yCarWorld']]), np.array([r_1['xCarWorld'], r_1['yCarWorld']]))

        observation = pd.DataFrame()
        observation.loc[0, 'xCarWorld'] = p['x']
        observation.loc[0, 'yCarWorld'] = p['y']
        observation.loc[0, 'nYawBody'] = p['yaw']
        observation.loc[0, 'nEngine'] = p['rpm']
        observation.loc[0, 'NGear'] = p['Gear']
        observation.loc[0, 'positionRho'] = rho
        observation.loc[0, 'positionTheta'] = theta
        observation.loc[0, 'positionReferenceX'] = r['xCarWorld']
        observation.loc[0, 'positionReferenceY'] = r['yCarWorld']
        observation.loc[0, 'positionRelativeX'] = rel_p[0]
        observation.loc[0, 'positionRelativeY'] = rel_p[1]
        observation.loc[0, 'referenceCurvature'] = ref_c
        observation.loc[0, 'actualCurvature'] = actual_c
        observation.loc[0, 'actualSpeedModule'] = v_actual_module
        observation.loc[0, 'speedDifferenceVectorModule'] = v_diff_module
        observation.loc[0, 'speedDifferenceOfModules'] = v_diff_of_modules
        observation.loc[0, 'actualAccelerationX'] = p['Acceleration_x']
        observation.loc[0, 'actualAccelerationY'] = p['Acceleration_y']
        observation.loc[0, 'referenceAccelerationX'] = r['Acceleration_x']
        observation.loc[0, 'referenceAccelerationY'] = r['Acceleration_y']
        observation.loc[0, 'accelerationDiffX'] = r['Acceleration_x'] - p['Acceleration_x']
        observation.loc[0, 'accelerationDiffY'] = r['Acceleration_y'] - p['Acceleration_y']
        observation.loc[0, 'prevaSteerWheel'] = prev_action[0]  #p_1['Steer']
        observation.loc[0, 'prevpBrakeF'] = prev_action[2]  #p_1['Brake']
        observation.loc[0, 'prevrThrottlePedal'] = prev_action[1]   #p_1['Throttle']

        """observation = dict()
        observation['time'] = p['curLapTime']
        observation['isReference'] = 0 #p['isReference']
        observation['is_partial'] = 0 #p['is_partial']
        observation['xCarWorld'] = p['x']
        observation['yCarWorld'] = p['y']
        observation['nYawBody'] = p['yaw']
        observation['nEngine'] = p['rpm']
        observation['NGear'] = p['Gear']
        observation['prevaSteerWheel'] = prev_action[0]  #p_1['Steer']
        observation['prevpBrakeF'] = prev_action[2]  #p_1['Brake']
        observation['prevrThrottlePedal'] = prev_action[1]   #p_1['Throttle']
        observation['positionRho'] = rho
        observation['positionTheta'] = theta
        observation['positionReferenceX'] = r['xCarWorld']
        observation['positionReferenceY'] = r['yCarWorld']
        observation['positionRelativeX'] = rel_p[0]
        observation['positionRelativeY'] = rel_p[1]
        observation['referenceCurvature'] = ref_c
        observation['actualCurvature'] = actual_c
        observation['actualSpeedModule'] = v_actual_module
        observation['speedDifferenceVectorModule'] = v_diff_module
        observation['speedDifferenceOfModules'] = v_diff_of_modules
        observation['actualAccelerationX'] = p['Acceleration_x']
        observation['actualAccelerationY'] = p['Acceleration_y']
        observation['referenceAccelerationX'] = r['Acceleration_x']
        observation['referenceAccelerationY'] = r['Acceleration_y']
        observation['accelerationDiffX'] = r['Acceleration_x'] - p['Acceleration_x']
        observation['accelerationDiffY'] = r['Acceleration_y'] - p['Acceleration_y']"""
        return observation.values, self.end_of_lap


    def act(self, ob, p_1, p_2, prev_action, reward):
        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        observation, nn = self.make_observaton(ob, p_1, p_2, prev_action)
        #print('observation:', observation)
        self.policy._actions = np.array(self.action_dispatcher.get_actions(observation))
        self.policy._n_actions = len(self.policy._actions)
        if self.policy._n_actions == 0:
            self.end_of_lap = 1
            return prev_action, self.end_of_lap, True
        self.policy.epsilon = 0.2
        observation = observation.reshape(1,-1)
        action = self.policy.sample_action(observation)
        #action = [[0,0,1,1]]
        return action[0], self.end_of_lap, False




class AgentMEAN(object):
    def __init__(self):
        self.car_df = pd.read_csv('trajectory/dataset.csv')
        self.car_tree = spatial.KDTree(list(zip(self.car_df['xCarWorld'], self.car_df['yCarWorld'])))
        laps = dict()
        trees = dict()
        for l in np.unique(self.car_df['NLap']):
            laps[l] = self.car_df.loc[self.car_df['NLap'] == l]
            trees[l] = spatial.KDTree(list(zip(laps[l]['xCarWorld'], laps[l]['yCarWorld'])))

    def act(self, ob):
        _, nns = self.car_tree.query([ob['x'], ob['y']], k = 10)
        steer = np.mean(self.car_df.loc[nns]['aSteerWheel'])
        throttle = np.mean(self.car_df.loc[nns]['rThrottlePedal'])
        brake = np.mean(self.car_df.loc[nns]['pBrakeF'])
        action = [steer,throttle*0.5,brake,0]
        return action
