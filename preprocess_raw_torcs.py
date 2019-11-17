import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_df = pd.read_csv('raw_torcs_data/forza_ow1.csv')
## Rename columns
raw_df.columns = ['curLapTime', 'Dist', 'Acceleration_x', 'Acceleration_y', 'Gear', 'rpm', 'speed_x', 'speed_y',
       'speed_z', 'dist_to_middle', 'trk_width', 'x', 'y', 'z', 'roll',
       'pitch', 'yaw', 'speedGlobalX', 'speedGlobalY', 'Steer', 'Throttle', 'Brake']
## Find rows with header instead of data and drop them
## Also drop the row before the "header" row because it is just to understand that the lap is finished   (no more required)
header_rows = raw_df.index[raw_df['curLapTime'] == 'time'].tolist()   # rows with headers
#header_rows = header_rows + [i-1 for i in header_rows]          # row before header (non serve più)
print("Header_rows: ", header_rows)
raw_df = raw_df.drop(header_rows)
raw_df = raw_df.astype(float)
## TORCS give speed in m/s, we want it in km/h
raw_df['speed_x'] *= 3.6
raw_df['speed_y'] *= 3.6
raw_df['speed_z'] *= 3.6
raw_df['rpm'] *= 10
## compute distance to middle of track
raw_df['dist_to_middle'] = 2*raw_df['dist_to_middle']/raw_df['trk_width'] #float dist_to_middle = 2*car->_trkPos.toMiddle/(car->_trkPos.seg->width);
raw_df = raw_df.drop(columns = 'trk_width')

## Now data are raw no more. Save to csv
raw_df.to_csv(path_or_buf = "car_trajectory.csv", index = False)

"""no_2ms = list()
for idx in range(1, raw_df.shape[0]):
    if (raw_df.iloc[idx]['curLapTime'] - raw_df.iloc[idx-1]['curLapTime']) > 0.021:
        print(raw_df.iloc[idx]['curLapTime'] - raw_df.iloc[idx-1]['curLapTime'])
        no_2ms.append(idx)

print("quandi?: ", len(no_2ms))"""
