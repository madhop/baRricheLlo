import numpy as np
import pandas as pd

from fqi.utils import *


def to_SARS(data, reward_function):
    # For each episode create sequence of samples
    episodes = {}
    for e in np.unique(data['NLap']):
        mask = data['NLap'] == e
        lap_df = data[mask]
        n_samples = np.count_nonzero(mask)
        # create timestamp column
        timestamp = pd.DataFrame({'t': np.zeros([n_samples - 1])})
        # Create NLap column
        nlap = pd.DataFrame({'NLap': np.ones([n_samples - 1]) * e}, dtype=int)
        # Add state and action
        state_action = lap_df[state_cols+action_cols].iloc[:-1].reset_index(drop=True)
        # Add reward
        reward = pd.DataFrame({'r': reward_function(lap_df)})
        # Add next state
        state_prime = lap_df[state_cols].iloc[1:].reset_index(drop=True)
        state_prime.rename(dict(zip(state_cols, state_prime_cols)), axis=1, inplace=True)
        # Add absorbing column
        absorbing = np.full([n_samples - 1, ], False)
        # If the lap is partial do not mark the last sample as the absorbing
        # otherwise do it
        #if ~(lap_df.is_partial.values[0]):
        if not (lap_df.is_partial.values[0]):
            absorbing[-1] = True
        absorbing = pd.DataFrame({'absorbing': absorbing}, dtype=bool)

        # episode starts: similarly to absoribing it is True for the first sample
        starts = np.full([n_samples - 1, ], False)
        starts[0] = True
        episode_starts = pd.DataFrame({'episode_starts': starts}, dtype=bool)

        episodes[e] = pd.concat((nlap, timestamp, state_action, reward, state_prime, absorbing, episode_starts), axis=1)

    sars = pd.concat(episodes)
    if reward_function.clipping:
        # In case of progress reward it is possible that some samples have too high or too low values due to out-track
        # position. Thus we remove all the samples with reward higher than +5 and lower than -20
        m, M = reward_function.clip_range
        clip_mask = (sars['r'] >= m) & (sars['r'] <= M)
        sars = sars[clip_mask]

    return sars
