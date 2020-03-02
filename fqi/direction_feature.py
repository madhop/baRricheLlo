import numpy as np


def create_direction_feature(data, x_col='xCarWorld', y_col='yCarWorld'):

    directions_x = list()
    directions_y = list()
    for e in np.unique(data['NLap']):
        mask = data['NLap'] == e
        lap_df = data[mask]
        n_samples = np.count_nonzero(mask)

        dir = lap_df[[x_col, y_col]].values[1:, :] - lap_df[[x_col, y_col]].values[:-1, :]
        dir = dir / np.linalg.norm(dir)
        dir = np.concatenate([dir[0, :][np.newaxis, :], dir])

        directions_x.extend(dir[:, 0])
        directions_y.extend(dir[:, 1])

    new_data = data.copy()
    new_data['direction_x'] = directions_x
    new_data['direction_y'] = directions_y

    return new_data