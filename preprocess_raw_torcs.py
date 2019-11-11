import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_df = pd.read_csv('raw_torcs_data/Test.csv', index_col = False)
first_rows = raw_df.index[raw_df['time'] == 'time'].tolist()
first_rows = first_rows + [i-1 for i in first_rows]
print("first_row: ", first_rows)
print("Prima: ", raw_df.iloc[first_rows])
raw_df = raw_df.drop(first_rows)
print("Dopo: ", raw_df.iloc[first_rows])
