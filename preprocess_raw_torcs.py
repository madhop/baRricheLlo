import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import compute_state_features as sf

"""
    Given raw data 100 Hz
    save 2 CSV files:
    - ref_trajectory.csv (x,y, alpha_step) 100 Hz
    - car_dataset.csv (all state features) 10 Hz
"""

track_length = 5780

## look for all possible CSV files and append all in the same DataFrame
raw_df = pd.DataFrame()
for d in range(1,32):
    for m in range(1,13):
        for y in range(2):
            try:
                df = pd.read_csv('raw_torcs_data/forza_ow'+str(2019+y)+'_'+str(m)+'_'+str(d)+'.csv', dtype='str') #forza_ow1_steering_w
                ## Rename columns
                df.columns = ['curLapTime', 'Dist', 'Acceleration_x', 'Acceleration_y', 'Gear', 'rpm', 'speed_x', 'speed_y',
                       'speed_z', 'dist_to_middle', 'trk_width', 'x', 'y', 'z', 'roll',
                       'pitch', 'yaw', 'speedGlobalX', 'speedGlobalY', 'Steer', 'Throttle', 'Brake']
                raw_df = raw_df.append(df, ignore_index = True)
            except:
                pass

raw_df['NLap'] = 0
raw_df['isReference'] = 0
raw_df['is_partial'] = 0

## Drop rows with header instead of data and drop them
header_rows = raw_df.index[raw_df['curLapTime'] == 'time'].tolist()   # rows with headers
#print("Header_rows: ", header_rows)
raw_df.drop(header_rows, inplace=True)
raw_df = raw_df.astype(float)
raw_df.index = np.arange(raw_df.shape[0])   # reset indexes

## divide laps
## find indeces of the beginning of lap
def add_row_at_top(df):
    """
    Add row of zeros at the beginning of the dataframe
    """
    df.loc[-1] = np.zeros(df.shape[1])  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index
    return df

lap_beginnings = ((raw_df['curLapTime'] - add_row_at_top(raw_df)['curLapTime'] < 0))    ## questo è un bel numero
lap_beginnings = [1] + raw_df.index[lap_beginnings].tolist() + [raw_df.shape[0]]    ## the last is fake, just to know where the last lap finishes
raw_df.drop(raw_df.tail(1).index,inplace=True)  ## drop last raw of zeros that is now useless
#print("lap_beginnings: ", lap_beginnings)

## for each row assign lap number, if it is reference trajectory and if it is partial lap
bestTime = np.inf
for i in range(len(lap_beginnings)-1):
    raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'NLap'] = i+1
    ## check if lap doesn't start from time zero
    if raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[0] > 1:
        t = raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'Dist'].iloc[0]/raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'speed_x'].iloc[0]
        raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'] = raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'] - raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[0] + t
    ## check if partial lap
    if raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'Dist'].iloc[-1] < track_length:
        raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'is_partial'] = 1
    ## check if best lap and set it as reference trajectory
    #print(i, 'time: ', raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[-1])
    if raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[-1] < bestTime and raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'is_partial'].iloc[0] != 1:
        bestTime = raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[-1]
        raw_df['isReference'] = 0
        raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'isReference'] = 1

#print("lap_beginnings: ", raw_df.loc[lap_beginnings])

## TORCS gives speed in m/s, we want it in km/h
raw_df['speed_x'] *= 3.6
raw_df['speed_y'] *= 3.6
raw_df['speed_z'] *= 3.6
raw_df['rpm'] *= 10
## compute distance to middle of track
raw_df['dist_to_middle'] = 2*raw_df['dist_to_middle']/raw_df['trk_width']
raw_df.drop(columns = 'trk_width', inplace=True)

## save reference trajectory in 100 Hz
#ref_df = raw_df[raw_df['isReference'] == 1].copy()
ref_df = pd.DataFrame()
for index, row in raw_df[raw_df['isReference'] == 1].iterrows():
    ref_df.loc[index, 'x'] = row['x']
    ref_df.loc[index, 'y'] = row['y']
    ## compute alpha step
    if index == raw_df[raw_df['isReference'] == 1].index[0] or index == raw_df[raw_df['isReference'] == 1].index[-1]:   ## first and last rows
        alpha_step = 0
    else:
        r = np.array([raw_df.loc[index-1, 'x'], raw_df.loc[index-1, 'y']])
        r1 = np.array([row['x'], row['y']])
        r2 = np.array([raw_df.loc[index+1, 'x'], raw_df.loc[index+1, 'y']])
        r2 = r2 - r1
        r1 = r1 - r
        alpha_step = sf.compute_alpha(r, r1, r2)
    ref_df.loc[index, 'alpha_step'] = alpha_step

## Export reference trajectory as CSV file
ref_df.to_csv(path_or_buf = "trajectory/test_ref_traj.csv", index = False)

## Downsampling actual trajectory
raw_df.index = np.arange(raw_df.shape[0])   # reset indexes
to_drop = raw_df.index[ raw_df.index % 10 != 0].tolist()   # rows with headers
raw_df.drop(to_drop, inplace=True)

## Compute state's features
raw_df.index = np.arange(raw_df.shape[0])   # reset indexes
last_ref = 0
for index, row in raw_df.iterrows():
    p = row
    p_1 = raw_df.iloc[index-1]
    p_2 = raw_df.iloc[index-2]
    nn = nn_ahead(np.array([p['x'], p['y']]), last_ref, ref_df)
    last_ref = nn
    r = ref_df.iloc[nn]
    r1 = ref_df.iloc[nn+1]
    r_1 = ref_df.iloc[nn-1]
    v_actual_module, v_ref_module, v_diff_module, v_diff_of_modules, v_angle = velocity_acceleration(np.array([p['speed_x'], p['speed_y']]), np.array([r['speed_x'], r['speed_y']]))
    tp = p['curLapTime'] - p_1['curLapTime'] # elapsed time between time t ant t-1
    tr = r['curLapTime'] - r_1['curLapTime'] # elapsed time between time t ant t-1
    ap = np.array([(p['speed_x'] - p_1['speed_x']) / tp, (p['speed_y'] - p_1['speed_y']) / tp])
    ar = np.array([(r['speed_x'] - r1['speed_x']) / tr, (r['speed_y'] - r1['speed_y']) / tr])
    a_actual_module, a_ref_module, a_diff_module, a_diff_of_modules, a_angle = velocity_acceleration(ap, ar)
    """if index == raw_df.index[0] or index == raw_df.index[-1]:   ## first and last rows
        pass
    else:"""
