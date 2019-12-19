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
        """
            from raw torcs observation to state features
        """
        p = ob
        nn = csf.nn_kdtree(np.array([p['x'], p['y']]), self.tree)
        # check if you are at the end of the lap
        if nn >= self.ref_df.shape[0]-1:
            nn = self.ref_df.shape[0]-2
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

        state_features = { 'xCarWorld' : p['x'], 'yCarWorld' : p['y'], 'nYawBody' : p['yaw'], 'nEngine' : p['rpm'], 'NGear' : p['Gear'],
                            'positionRho' : rho, 'positionTheta' : theta, 'positionReferenceX' : r['xCarWorld'], 'positionReferenceY' : r['yCarWorld'],
                            'positionRelativeX' : rel_p[0], 'positionRelativeY' : rel_p[1], 'referenceCurvature' : ref_c, 'actualCurvature' : actual_c,
                            'actualSpeedModule' : v_actual_module, 'speedDifferenceVectorModule' : v_diff_module, 'speedDifferenceOfModules' : v_diff_of_modules,
                            'actualAccelerationX' : p['Acceleration_x'], 'actualAccelerationY' : p['Acceleration_y'],
                            'referenceAccelerationX' : r['Acceleration_x'], 'referenceAccelerationY' : r['Acceleration_y'],
                            'accelerationDiffX' : r['Acceleration_x'] - p['Acceleration_x'], 'accelerationDiffY' : r['Acceleration_y'] - p['Acceleration_y'],
                            'prevaSteerWheel' : prev_action[0], 'prevpBrakeF' : prev_action[2], 'prevrThrottlePedal' : prev_action[1] }

        observation = pd.DataFrame()
        for k in state_cols:
            observation.loc[0, k] = state_features[k]


        return observation.values


    def act(self, ob, p_1, p_2, prev_action, reward):
        observation = self.make_observaton(ob, p_1, p_2, prev_action)
        self.policy._actions = np.array(self.action_dispatcher.get_actions(observation))
        self.policy._n_actions = len(self.policy._actions)
        if self.policy._n_actions == 0:
            return []
        self.policy.epsilon = 0.2
        observation = observation.reshape(1,-1)
        action = self.policy.sample_action(observation)
        gear = 0    # fake gear, automatic gear shift
        action = np.append(action, [gear])
        #action = [[0,0,1,0]]   # fake action, staight, full gass
        return action




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
