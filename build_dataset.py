import compute_state_features as sf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    From torcs data, compute state features and append to Dataset.csv
"""

def buildDataset(file_name = 'preprocessed_torcs_algo'):
    plot_coords = False
    ref_file_name = 'trajectory/ref_traj.csv'

    ref_df = pd.read_csv(ref_file_name) # reference trajectory
    ref_df.columns = ['curLapTime', 'Acceleration_x', 'Acceleration_y', 'speed_x', 'speed_y', 'x', 'y', 'alpha_step']
    car_df = pd.read_csv('raw_torcs_data/' + file_name + '.csv') # car trajectory
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
            if p['NLap'] != last_lap:
                print('lap', p['NLap'])
                last_ref = 0
                last_lap = p['NLap']
            else:
                last_ref = nn
            r = ref_df.iloc[nn]
            r1 = ref_df.iloc[nn+1]
            r_1 = ref_df.iloc[nn-1]
            r_2 = ref_df.iloc[nn-2]
            v_actual_module, v_ref_module, v_diff_module, v_diff_of_modules, v_angle = sf.velocity_acceleration(np.array([p['speed_x'], p['speed_y']]), np.array([r['speed_x'], r['speed_y']]))
            ap = np.array([p['Acceleration_x'], p['Acceleration_y']])
            ar = np.array([r['Acceleration_x'], r['Acceleration_y']])
            a_actual_module, a_ref_module, a_diff_module, a_diff_of_modules, a_angle = sf.velocity_acceleration(ap, ar)
            rel_p, rho, theta = sf.position(np.array([r['x'], r['y']]), np.array([r1['x'], r1['y']]), np.array([p['x'], p['y']]))
            actual_c = sf.curvature(np.array([p['x'], p['y']]), np.array([p_1['x'], p_1['y']]), np.array([p_2['x'], p_2['y']]))
            ref_c = sf.curvature(np.array([r1['x'], r1['y']]), np.array([r['x'], r['y']]), np.array([r_1['x'], r_1['y']]))

            if index > 10 and plot_coords:
                print('rel_p:', rel_p)
                rel_p_2,_,_ = sf.position(np.array([r_2['x'], r_2['y']]), np.array([r_1['x'], r_1['y']]), np.array([p['x'], p['y']]))
                print('rel_p_2:', rel_p_2)
                plt.plot([ref_df.iloc[nn-i-1]['x'] for i in range(5)] + [r_1['x'], r['x'], r1['x']],[ref_df.iloc[nn-i]['y'] for i in range(5)] + [r_1['y'], r['y'], r1['y']], 'o')
                plt.plot([p['x']],[p['y']], '*')
                plt.plot([r['x']],[r['y']], '*')
                plt.show()


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
            actual_df.loc[index, 'positionRelativeX'] = rel_p[0]
            actual_df.loc[index, 'positionRelativeY'] = rel_p[1]
            actual_df.loc[index, 'referenceCurvature'] = ref_c
            actual_df.loc[index, 'actualCurvature'] = actual_c
            actual_df.loc[index, 'actualSpeedModule'] = v_actual_module
            actual_df.loc[index, 'speedDifferenceVectorModule'] = v_diff_module
            actual_df.loc[index, 'speedDifferenceOfModules'] = v_diff_of_modules
            actual_df.loc[index, 'actualAccelerationX'] = p['Acceleration_x']
            actual_df.loc[index, 'actualAccelerationY'] = p['Acceleration_y']
            actual_df.loc[index, 'referenceAccelerationX'] = r['Acceleration_x']
            actual_df.loc[index, 'referenceAccelerationY'] = r['Acceleration_y']
            actual_df.loc[index, 'accelerationDiffX'] = r['Acceleration_x'] - p['Acceleration_x']
            actual_df.loc[index, 'accelerationDiffY'] = r['Acceleration_y'] - p['Acceleration_y']
            actual_df.loc[index, 'aSteerWheel'] =  p['Steer']
            actual_df.loc[index, 'pBrakeF'] =  p['Brake']
            actual_df.loc[index, 'rThrottlePedal'] =  p['Throttle']

    #actual_df.to_csv(path_or_buf = "trajectory/dataset_70_laps.csv", index = False)
    actual_df.to_csv(path_or_buf = "trajectory/dataset.csv", mode = 'a', index = False, header = False)


if __name__ == "__main__":
    buildDataset()
