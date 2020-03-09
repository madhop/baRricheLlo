from stable_baselines.gail.dataset.dataset import ExpertDataset
import pandas as pd
import numpy as np
from fqi.direction_feature import create_direction_feature
from fqi.sars_creator import to_SARS


def create_expert_dataset(demonstrations_path, reward_function, state_cols, action_cols, batch_size, train_fraction):
    """Create demonstrations dataset as instance of the class ExpertDataset.

    :param demonstrations_path: (str) the of the csv file containing demonstrations
    :param reward_function: (Reward_function) instance of the class Reward_function used to
            compoute the reward
    :param state_cols: (list) list containing the features used as state
    :param action_cols (list) list containing the actions
    :param batch_size: (int) batch size the dataset
    :param train_fraction (float) percentage of training set with respect to validation set
    :return: (ExpertDataset) instance of the class containing demonstrations
    """

    demos = pd.read_csv(demonstrations_path)
    if ('direction_x' not in demos.columns) or\
            ('direction_y' not in demos.columns):
        demos = create_direction_feature(demos)

    demos = to_SARS(demos, reward_function, state_cols)
    demos = demos.reset_index(drop=True)

    starts_index = demos[demos['episode_starts']].index
    episode_returns = []

    for i in range(len(starts_index) - 1):
        episode_returns.append(demos.loc[starts_index[i]:starts_index[i+1]]['r'].su())
    episode_returns.append(demos.loc[starts_index[-1]:]['r'].sum())

    demos_dict = dict()
    demos_dict['actions'] = demos[action_cols].values
    demos_dict['obs'] = demos[state_cols].values
    demos_dict['rewards'] = demos['r'].values
    demos_dict['episode_starts'] = demos['episode_starts']
    demos_dict['episode_returns'] = np.array(episode_returns)

    return ExpertDataset(traj_data=demos_dict, batch_size=batch_size, train_fraction=train_fraction)
