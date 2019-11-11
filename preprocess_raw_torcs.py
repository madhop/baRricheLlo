import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_df = pd.read_csv('raw_torcs_data/Test.csv', index_col = False)

# Find rows with header instead of data and drop them
# Also drop the row before the "header" row because it is just to understand that the lap is finished
header_rows = raw_df.index[raw_df['time'] == 'time'].tolist()   # rows with headers
header_rows = header_rows + [i-1 for i in header_rows]          # row before headers
raw_df = raw_df.drop(header_rows)
