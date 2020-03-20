import pandas as pd
import numpy as np
from numpy.matlib import repmat
import os
from sklearn.neighbors import KDTree
import compute_state_features as csf


def preproc_raw_torcs(data_dir, out_dir, out_ref, track_length=5780, subsampling_step=10):
    """
    Given the TORCS raw data it adds lap number, trackPos feature, perform subsampling and define the reference
    :param data_dir: (str) path containing raw TORCS data
    :param out_dir: (str) path where to save processed data
    :param out_ref: (str) path where to save reference trajectory
    :param track_length: (int) length of the track
    :param subsampling_step: (int) step of subsampling
    """

    files = list(filter(lambda x: x[-4:] == '.csv', os.listdir(data_dir)))
    raw_data = [pd.read_csv(os.path.join(data_dir, f)) for f in files]
    raw_data = pd.concat(raw_data, axis=0).reset_index(drop=True)
    # Rename columns
    raw_data.columns = ['curLapTime', 'Dist', 'Acceleration_x', 'Acceleration_y', 'Gear', 'rpm', 'speed_x', 'speed_y',
                        'speed_z', 'dist_to_middle', 'trk_width', 'x', 'y', 'z', 'roll',
                        'pitch', 'yaw', 'speedGlobalX', 'speedGlobalY', 'Steer', 'Throttle', 'Brake']

    # Find the start and the end of each lap and the corresponding lap time
    lap_beginning = (raw_data['curLapTime'] - raw_data['curLapTime'].shift(periods=1, fill_value=0)) < 0
    # set first sample to be lap beginning
    lap_beginning[0] = True
    lap_end = (raw_data['curLapTime'] - raw_data['curLapTime'].shift(periods=-1, fill_value=0)) > 0
    lap_time = raw_data['curLapTime'][lap_end].values - raw_data['curLapTime'][lap_beginning].values
    lap_steps = raw_data.index[lap_end] - raw_data.index[lap_beginning] + 1
    # add NLap column
    lap_list = np.array(list(range(len(lap_steps))))
    NLap = np.concatenate([np.matlib.repmat(i, lap_steps[i], 1).ravel() for i in lap_list])

    raw_data['NLap'] = NLap

    # Remove laps with time greater than 90
    time_limit = 90.0
    good_lap_mask = lap_time <= time_limit
    good_laps = lap_list[good_lap_mask]
    raw_data = raw_data[raw_data['NLap'].isin(good_laps)]

    # filter all the structures
    lap_steps = lap_steps[good_lap_mask]
    lap_time = lap_time[good_lap_mask]
    lap_beginning = (raw_data['curLapTime'] - raw_data['curLapTime'].shift(periods=1, fill_value=0)) < 0
    # set first sample to be lap beginning
    lap_beginning[0] = True
    lap_end = (raw_data['curLapTime'] - raw_data['curLapTime'].shift(periods=-1, fill_value=0)) > 0

    # the lap is partial if the Dist of the last sample is lower than the track_length
    partial = raw_data['Dist'][lap_end].values < track_length
    partial_ext = np.concatenate([np.matlib.repmat(partial[i], lap_steps[i], 1).ravel() for i in range(len(lap_steps))])
    raw_data['is_partial'] = partial_ext

    # find the reference trajectory
    times_to_search = lap_time.copy()
    times_to_search[partial] = np.inf
    best_lap = good_laps[np.argmin(times_to_search)]

    is_reference = np.zeros(raw_data.shape[0], dtype=bool)
    is_reference[raw_data.NLap == best_lap] = True
    raw_data['isReference'] = is_reference

    # Feature transformation

    # Transform speed into km/h
    raw_data['speed_x'] *= 3.6
    raw_data['speed_y'] *= 3.6
    raw_data['speed_z'] *= 3.6
    raw_data['rpm'] *= 10

    # compute trackPos i.e., the distance to middle of the track
    raw_data['trackPos'] = 2 * raw_data['dist_to_middle'] / raw_data['trk_width']
    raw_data.drop(columns='trk_width', inplace=True)

    # Save reference trajectory
    ref_df = raw_data[raw_data['isReference']].copy()
    ref_df = ref_df.reset_index(drop=True)
    ref_df.rename(columns={'x': 'xCarWorld', 'y': 'yCarWorld'}, inplace=True)
    ref_df.to_csv(os.path.join(out_ref, 'ref_traj.csv'), index=False)

    # Perform subsampling
    sampled_raw_data = [raw_data[raw_data.NLap == lap][::subsampling_step] for lap in set(raw_data.NLap)]
    sampled_raw_data = pd.concat(sampled_raw_data, axis=0).reset_index(drop=True)
    sampled_raw_data.to_csv(os.path.join(out_dir, 'preprocessed_torcs.csv'), index=False)


def extract_features(raw_file, ref_file, out_dir):
    """
    Given preprocessed TORCS data it extracts relative features wrt the reference trajectory.
    :param raw_file: (str) path where to load the data
    :param ref_file: (str) path where the reference trajectory is
    :param out_dir: (str) path where to save the new data
    """

    raw_data = pd.read_csv(raw_file)
    ref_df = pd.read_csv(ref_file)
    dir_ = ref_df[['xCarWorld', 'yCarWorld']].values[1:, :] - ref_df[['xCarWorld', 'yCarWorld']].values[:-1, :]
    dir_ = dir_ / np.linalg.norm(dir_)
    dir_ = np.concatenate([dir_[0, :].reshape(1, 2), dir_], axis=0)
    ref_df['direction_x'] = dir_[:, 0]
    ref_df['direction_y'] = dir_[:, 1]

    kdt = KDTree(ref_df[['xCarWorld', 'yCarWorld']].values)

    data = []
    for lap in set(raw_data.NLap):
        lap_df = raw_data[raw_data.NLap == lap].copy()
        data_lap = pd.DataFrame()
        # base features
        data_lap['NLap'] = lap_df.NLap
        data_lap['is_partial'] = lap_df.is_partial
        data_lap['isReference'] = lap_df.isReference
        data_lap['trackPos'] = lap_df.trackPos
        data_lap['time'] = lap_df.curLapTime
        data_lap['NGear'] = lap_df.Gear
        data_lap['nEngine'] = lap_df.rpm
        data_lap['xCarWorld'] = lap_df.x
        data_lap['yCarWorld'] = lap_df.y
        data_lap['nYawBody'] = lap_df.yaw
        data_lap['aSteerWheel'] = lap_df.Steer
        data_lap['rThrottlePedal'] = lap_df.Throttle
        data_lap['pBrakeF'] = lap_df.Brake
        data_lap['acceleration_x'] = lap_df.Acceleration_x
        data_lap['acceleration_y'] = lap_df.Acceleration_y
        data_lap['speed_x'] = lap_df.speed_x
        data_lap['speed_y'] = lap_df.speed_y
        # previous actions
        prev_steer = lap_df.shift(periods=1).Steer.values
        prev_steer[0] = prev_steer[1]
        data_lap['prevaSteerWheel'] = prev_steer
        prev_brake = lap_df.shift(periods=1).Brake.values
        prev_brake[0] = prev_brake[1]
        data_lap['prevpBrakeF'] = prev_brake
        prev_throttle = lap_df.shift(periods=1).Throttle.values
        prev_throttle[0] = prev_throttle[1]
        data_lap['prevrThrottlePedal'] = prev_throttle

        # direction
        dir_ = data_lap[['xCarWorld', 'yCarWorld']].values[1:, :] - data_lap[['xCarWorld', 'yCarWorld']].values[:-1, :]
        dir_ = dir_ / np.linalg.norm(dir_)
        dir_ = np.concatenate([dir_[0, :].reshape(1, 2), dir_], axis=0)
        data_lap['direction_x'] = dir_[:, 0]
        data_lap['direction_y'] = dir_[:, 1]

        # relative features:
        # for each point find the nearest of the reference
        nn_ref = kdt.query(data_lap[['xCarWorld', 'yCarWorld']], return_distance=False).ravel()
        nn_ref_next = nn_ref + 1
        nn_ref_next[nn_ref_next == ref_df.shape[0]] = 0
        # delta speed
        data_lap['delta_speed_x'] = ref_df['speed_x'].values[nn_ref] - data_lap['speed_x']
        data_lap['delta_speed_y'] = ref_df['speed_y'].values[nn_ref] - data_lap['speed_y']
        # delta acceleration
        data_lap['delta_acc_x'] = ref_df['Acceleration_x'].values[nn_ref] - data_lap['acceleration_x']
        data_lap['delta_acc_y'] = ref_df['Acceleration_y'].values[nn_ref] - data_lap['acceleration_y']
        # delta direction
        data_lap['delta_direction_x'] = ref_df['direction_x'].values[nn_ref] - data_lap['direction_x']
        data_lap['delta_direction_y'] = ref_df['direction_y'].values[nn_ref] - data_lap['direction_y']
        # polar coordinates with respect reference trajectory
        p = data_lap[['xCarWorld', 'yCarWorld']].values
        r = ref_df[['xCarWorld', 'yCarWorld']].values[nn_ref]
        r_next = ref_df[['xCarWorld', 'yCarWorld']].values[nn_ref_next]
        _, rho, theta = csf.vector_position(r, r_next, p)
        data_lap['positionRho'] = rho
        data_lap['positionTheta'] = theta

        data.append(data_lap)

    data = pd.concat(data, axis=0).reset_index(drop=True)
    data.to_csv(os.path.join(out_dir, 'demonstrations.csv'), index=False)


def torcs_observation_to_state(obs, obs_1, obs_2, prev_action, state_cols, ref_df, ref_kdt):
    """
    Transform a TORCS observation to state array.
    :param obs:  (dict) observation at time t
    :param obs_1: (dict) observation at time t-1
    :param obs_2: (dict) observation at time t-2
    :param prev_action: (list) previous action
    :param state_cols: (list) list of states features to use
    :param ref_df: (DataFrame) reference trajectory
    :param ref_kdt: (KDTree) KDTree for fast search of reference samples
    :return: state vector (np.array) state vector
    """
    nn_ref = ref_kdt.query(np.array([obs['x'], obs['y']]).reshape(1, 2), return_distance=False).ravel()
    prev_nn_ref = nn_ref - 1 if nn_ref > 0 else ref_df.shape[0] - 1
    next_nn_ref = nn_ref + 1 if nn_ref < (ref_df.shape[0] - 1) else 0
    ref_direction = ref_df[['xCarWorld', 'yCarWorld']].values[nn_ref] -\
                    ref_df[['xCarWorld', 'yCarWorld']].values[prev_nn_ref]
    ref_direction = ref_direction / np.linalg.norm(ref_direction)
    ref_direction = ref_direction.ravel()

    state = dict()
    state['trackPos'] = obs['trackPos']
    state['NGear'] = obs['Gear']
    state['xCarWorld'] = obs['x']
    state['yCarWorld'] = obs['y']
    state['nEngine'] = obs['rpm']
    state['nYawBody'] = obs['yaw']
    state['speed_x'] = obs['speed_x']
    state['speed_y'] = obs['speed_y']
    state['acceleration_x'] = obs['Acceleration_x']
    state['acceleration_y'] = obs['Acceleration_y']
    state['prevaSteerWheel'] = prev_action[0]
    state['prevpBrakeF'] = prev_action[1]
    state['prevrThrottlePedal'] = prev_action[2]

    # direction
    dir_ = np.array([obs['x'], obs['y']]) - np.array([obs_1['x'], obs_1['y']])
    dir_ = dir_ / np.linalg.norm(dir_)
    state['direction_x'] = dir_[0]
    state['direction_y'] = dir_[1]

    # relative features
    state['delta_speed_x'] = ref_df['speed_x'].values[nn_ref] - obs['speed_x']
    state['delta_speed_y'] = ref_df['speed_y'].values[nn_ref] - obs['speed_y']
    state['delta_acc_x'] = ref_df['Acceleration_x'].values[nn_ref] - obs['Acceleration_x']
    state['delta_acc_y'] = ref_df['Acceleration_y'].values[nn_ref] - obs['Acceleration_y']
    state['delta_direction_x'] = ref_direction[0] - dir_[0]
    state['delta_direction_y'] = ref_direction[1] - dir_[1]
    p = np.array([obs['x'], obs['y']])
    r = ref_df[['xCarWorld', 'yCarWorld']].values[nn_ref].ravel()
    r_next = ref_df[['xCarWorld', 'yCarWorld']].values[next_nn_ref].ravel()
    _, rho, theta = csf.position(r, r_next, p)
    state['positionRho'] = rho
    state['positionTheta'] = theta

    return np.array([state[k] for k in state_cols])
