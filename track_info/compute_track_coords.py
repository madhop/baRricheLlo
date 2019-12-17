import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torcs_data = pd.read_csv('track_data/track_data_monza.csv')
"""torcs_data.columns = ['angle', 'tgAngle', 'segAngle', 'curLapTime', 'lastLapTime', 'speed_x',
       'speed_y', 'speed_z', 'distFromStart', 'trackPos', 'x', 'y', 'z', 'yaw']"""

print('torcs_data.shape', torcs_data.shape)

def distance(x, y, x1, y1):
    dist = np.sqrt((x-x1)**2 + (y-y1)**2)
    return dist

out_data = {'x':[], 'y':[],  'x1':[], 'y1':[], 'x2':[], 'y2':[], 'width':[]}

for index, row in torcs_data.iterrows():
    #print(row[['x','y']])
    out_data['x'].append(row['x'])
    out_data['y'].append(row['y'])
    d = 5.5 * row['trackPos']
    theta = row['segAngle']
    x1 = row['x'] - (d + 5.5)* np.cos(theta)
    y1 = row['y'] - (d + 5.5)* np.sin(theta)
    out_data['x1'].append(x1)
    out_data['y1'].append(y1)
    x2 = row['x'] - (d - 5.5)* np.cos(theta)
    y2 = row['y'] - (d - 5.5)* np.sin(theta)
    out_data['x2'].append(x2)
    out_data['y2'].append(y2)
    track_width = distance(x1, y1, x2, y2)
    out_data['width'].append(track_width)

plt.plot(out_data['x'],out_data['y'], color='b')
for index, row in torcs_data.iterrows():
    if index%20 == 0:
        plt.text(row['x'], row['y'], str(index))

plt.text(out_data['x'][0],out_data['y'][0], str(10))
plt.plot(out_data['x1'],out_data['y1'], color = 'r')
plt.plot(out_data['x2'],out_data['y2'], color = 'g')



i = 660
print('width at', i, distance(out_data['x1'][i], out_data['y1'][i], out_data['x2'][i], out_data['y2'][i]))
plt.plot([out_data['x1'][i],out_data['x2'][i]],[out_data['y1'][i],out_data['y2'][i]],'k-')
i = 661
print('width at', i, distance(out_data['x1'][i], out_data['y1'][i], out_data['x2'][i], out_data['y2'][i]))
plt.plot([out_data['x1'][i],out_data['x2'][i]],[out_data['y1'][i],out_data['y2'][i]],'k-')
i = 662
print('width at', i, distance(out_data['x1'][i], out_data['y1'][i], out_data['x2'][i], out_data['y2'][i]))
plt.plot([out_data['x1'][i],out_data['x2'][i]],[out_data['y1'][i],out_data['y2'][i]],'k-')
i = 663
print('width at', i, distance(out_data['x1'][i], out_data['y1'][i], out_data['x2'][i], out_data['y2'][i]))
plt.plot([out_data['x1'][i],out_data['x2'][i]],[out_data['y1'][i],out_data['y2'][i]],'k-')

plt.show()
