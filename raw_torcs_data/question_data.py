import pandas as pd

df = pd.read_csv('forza_ow1_10ms.csv')

df.columns = ['curLapTime', 'Dist', 'Acceleration_x', 'Acceleration_y', 'Gear', 'rpm', 'speed_x', 'speed_y',
       'speed_z', 'dist_to_middle', 'trk_width', 'x', 'y', 'z', 'roll',
       'pitch', 'yaw', 'speedGlobalX', 'speedGlobalY', 'Steer', 'Throttle', 'Brake']

"""count = 0
for i in range(1,df.shape[0]):
	d = df.iloc[i]['time'] - df.iloc[i-1]['time']
	if d >= 0.0099:
		count += 1
		print(i, ': ', d)

print('count: ', count)
print('df.shape: ', df.shape)"""

steer = df.index[df['Steer'] > 0.6].tolist()   # rows with headers
print("steer list: ", steer)
