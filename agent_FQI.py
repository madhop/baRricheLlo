import numpy as np
import compute_state_features as csf
import pandas as pd
from scipy import spatial

class AgentFQI(object):
    def __init__(self, ref_df):
        self.ref_df = ref_df
        self.tree = spatial.KDTree(list(zip(ref_df['x'], ref_df['y'])))
        self.end_of_lap = False


    def make_observaton(self, ob, p_1, p_2, prev_action):
        p = ob
        nn = csf.nn_kdtree(np.array([p['x'], p['y']]), self.tree)
        if nn >= self.ref_df.shape[0]-1:
            print('sei alla fine del giro')
            nn = self.ref_df.shape[0]-2
            self.end_of_lap = True
        r = self.ref_df.iloc[nn]
        r1 = self.ref_df.iloc[nn+1]
        r_1 = self.ref_df.iloc[nn-1]

        v_actual_module, v_ref_module, v_diff_module, v_diff_of_modules, v_angle = csf.velocity_acceleration(np.array([p['speed_x'], p['speed_y']]), np.array([r['speed_x'], r['speed_y']]))
        ap = np.array([p['Acceleration_x'], p['Acceleration_y']])
        ar = np.array([r['Acceleration_x'], r['Acceleration_y']])
        a_actual_module, a_ref_module, a_diff_module, a_diff_of_modules, a_angle = csf.velocity_acceleration(ap, ar)
        rel_p, rho, theta = csf.position(np.array([r['x'], r['y']]), np.array([r1['x'], r1['y']]), np.array([p['x'], p['y']]))
        actual_c = csf.curvature(np.array([p['x'], p['y']]), np.array([p_1['x'], p_1['y']]), np.array([p_2['x'], p_2['y']]))
        ref_c = csf.curvature(np.array([r1['x'], r1['y']]), np.array([r['x'], r['y']]), np.array([r_1['x'], r_1['y']]))

        observation = dict()
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
        observation['positionReferenceX'] = r['x']
        observation['positionReferenceY'] = r['y']
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
        observation['accelerationDiffY'] = r['Acceleration_y'] - p['Acceleration_y']
        return observation, self.end_of_lap


    def act(self, ob, p_1, p_2, prev_action, reward):
        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        observation, nn = self.make_observaton(ob, p_1, p_2, prev_action)
        action = #[0,1,0,0]
        return action, self.end_of_lap


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
