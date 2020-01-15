import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import compute_state_features as sf

def preprocess_raw_torcs(output_name = "preprocessed_torcs_algo"):
    track_length = 5780

    dataset = pd.read_csv('trajectory/dataset.csv')
    n_laps = dataset.tail(1)['NLap'].values
    #print('n_laps:', n_laps)

    raw_df = pd.read_csv('raw_torcs_data/raw_data_algo.csv', dtype='str')
    raw_df['NLap'] = 0
    raw_df['isReference'] = 0
    raw_df['is_partial'] = 0

    ## Drop rows with header instead of data and drop them
    header_rows = raw_df.index[raw_df['Acceleration_x'] == 'Acceleration_x'].tolist()   # rows with headers
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

    lap_beginnings = ((raw_df['curLapTime'] - add_row_at_top(raw_df)['curLapTime'] < 0))
    lap_beginnings = [1] + raw_df.index[lap_beginnings].tolist() + [raw_df.shape[0]]    ## the last is fake, just to know where the last lap finishes
    raw_df.drop(raw_df.tail(1).index,inplace=True)  ## drop last raw of zeros that is now useless
    #print("lap_beginnings: ", lap_beginnings)

    ## Drop really bad laps
    #print('raw_df.shape:', raw_df.shape)
    slow_laps = list()
    for i, lap_beg in enumerate(lap_beginnings[:-1]):
        if raw_df.loc[lap_beginnings[i+1]-1, 'curLapTime'] > 90:
            print(i, lap_beg, 'troppo lento', raw_df.loc[lap_beginnings[i+1]-1, 'curLapTime'])
            raw_df.drop(np.arange(lap_beg, lap_beginnings[i+1]), inplace=True)
            slow_laps = slow_laps + [lap_beg]

    for i in slow_laps:
        lap_beginnings.remove(i)

    ## for each row assign lap number and if it is partial lap
    bestTime = np.inf
    for i, lap_beg in enumerate(lap_beginnings[:-1]):
    #for i in range(len(lap_beginnings)-1):
        raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'NLap'] = i+1+n_laps
        ## check if lap doesn't start from time zero
        if raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'curLapTime'].iloc[0] > 1:
            t = raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'distFromStart'].iloc[0]/raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'speed_x'].iloc[0]
            raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'curLapTime'] = raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'curLapTime'] - raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'curLapTime'].iloc[0] + t
        ## check if partial lap
        if raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'distFromStart'].iloc[-1] < track_length:
            raw_df.loc[lap_beg:lap_beginnings[i+1]-1, 'is_partial'] = 1

    print('# laps:', i+1)

    raw_df.to_csv(path_or_buf = "raw_torcs_data/" + output_name + ".csv", index = False)
