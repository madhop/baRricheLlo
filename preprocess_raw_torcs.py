import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
    Given raw data 100 Hz
    save 2 CSV files:
    - ref_trajectory.csv (x,y, alpha_step) 100 Hz
    - car_dataset.csv (all state features) 10 Hz
"""

track_length = 5780

raw_df  =pd.DataFrame()
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

## Find rows with header instead of data and drop them
header_rows = raw_df.index[raw_df['curLapTime'] == 'time'].tolist()   # rows with headers
print("Header_rows: ", header_rows)
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


raw_df['NLap'] = 0
raw_df['isReference'] = 0
raw_df['is_partial'] = 0

#print("raw_df.shape: ", raw_df.shape)
lap_beginnings = ((raw_df['curLapTime'] - add_row_at_top(raw_df)['curLapTime'] < 0))
lap_beginnings = [1] + raw_df.index[lap_beginnings].tolist() + [raw_df.shape[0]]    ## the last is fake, just to know where the last lap finishes
raw_df.drop(raw_df.tail(1).index,inplace=True)
print("lap_beginnings: ", lap_beginnings)

bestTime = np.inf
for i in range(len(lap_beginnings)-1):
    raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'NLap'] = i+1
    ## check if lap doesn't start from time zero
    if raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[0] > 1:
        #print('***** NO START FROM ZERO ******')
        t = raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'Dist'].iloc[0]/raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'speed_x'].iloc[0]
        raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'] = raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'] - raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[0] + t
    ## check if partial lap
    if raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'Dist'].iloc[-1] < track_length:
        #print('***** PARTIAL LAP ******')
        raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'is_partial'] = 1
    ## check if best lap and set it as reference trajectory
    print(i, 'time: ', raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[-1])
    if raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'curLapTime'].iloc[-1] < bestTime and raw_df.loc[lap_beginnings[i]:lap_beginnings[i+1]-1, 'is_partial'].iloc[0] != 1:
        #print('***** BEST LAP (for the moment) ******')
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
raw_df['dist_to_middle'] = 2*raw_df['dist_to_middle']/raw_df['trk_width'] #float dist_to_middle = 2*car->_trkPos.toMiddle/(car->_trkPos.seg->width);
raw_df.drop(columns = 'trk_width', inplace=True)


ref_df = raw_df[raw_df['isReference'] == 1].copy()
## Now data are raw no more. Save to csv
#raw_df.to_csv(path_or_buf = "trajectory/forza_ow1_steering_w.csv", index = False)
