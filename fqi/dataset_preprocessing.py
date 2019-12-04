import pandas as pd
import numpy as np
import itertools 
from sklearn.neighbors import KDTree
from fqi.utils import *
import sys
import os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '..'))
from trlib.utilities.ActionDispatcher import *
from feature_extraction.rt_normalization import normalize
from sklearn.preprocessing import MinMaxScaler

reference_reward = 10

def prepare_dataset(data_path, ref_path, reward_function='progress', delta_t=0.1):
    """
    From csv
    
    Input:
    data_path -- string containing the path of the file containing the simulations
    ref_path -- string containing the path of the reference trajectory file
    
    """
    # load episodes (laps)
    df = pd.read_csv(data_path, dtype={'isReference': bool, 'is_partial': bool})

    # Load the reference trajectory
    rt = pd.read_csv(ref_path)

    return prepare(df, rt, reward_function, delta_t)

def prepare(df, rt, reward_function='progress', delta_t=0.1):
    """
    From dataframe
    
    Given a set of laps on one track, it returns the dataset
    as a sequence of steps in the form:
    timestamp | state | action | reward | next_state | absorbing

    Keyword arguments:
    reward_function -- 'delta_t': r(s_t,a_t) = - delta_t
                       'progress': the reward is equal to the number of forward steps
    delta_t -- delta time
    
    Output:
    dataset -- DataFrame with each sample as row
    """
    
    # load episodes (laps)
    #df = pd.read_csv(data_path, dtype={'isReference': bool, 'is_partial': bool})

    # Load the reference trajectory
    #rt = pd.read_csv(ref_path)
    if reward_function == 'spatial_progress':
        rt, step_dist = normalize(rt)
    alpha_step = rt['alpha_step'].values
    # Compute the KDTree to a quick search of the nearest-neighbor
    ref_xy = np.array(list(zip(rt['xCarWorld'].values, rt['yCarWorld'].values)))
    kdtree = KDTree(ref_xy)

    # For each episode create sequence of samples
    episodes = {}
    for e in np.unique(df['NLap']):

        mask = df['NLap'] == e

        lap_df = df[mask]
        n_samples = np.count_nonzero(mask)

        # create timestamp column
        timestamp_df = pd.DataFrame({'t': np.zeros([n_samples-1])})
        
        # Create NLap column
        nlap_df = pd.DataFrame({'NLap': np.ones([n_samples-1]) * e})
        
        # Add state and previous actions
        state = {}
        for s in state_cols:
            state[s] = np.array(lap_df[s])[0:-1]

        state_df = pd.DataFrame(state)

        # Add action
        action = {}
        for a in action_cols:
            action[a] = np.array(lap_df[a][0:-1]).reshape(n_samples-1)
        action_df = pd.DataFrame(action)

        # Add reward

        if reward_function == 'delta_t':

            reward = np.ones([n_samples-1],) * -delta_t

        elif reward_function == 'progress' or reward_function == 'spatial_progress':
            
            # The reward is the number of steps between s and s'            
            # Given a point s, we compute the x' coordinate with respect to its
            # nearest reference point (considering rotation and translation).
            # if x' > 0 then we consider the next reference point
            # if x' <= 0 then we consider the current reference point
            # The same procedure is done for the point s'
            # The reward is equal to the difference of the progress number of
            # the two reference points
            
            # Find the nearest point ids for each point in the simulation
            points_xy = list(zip(lap_df['xCarWorld'].values, lap_df['yCarWorld'].values))
            
            _, ref_id = kdtree.query(points_xy)
            ref_id = ref_id.squeeze()
            # compute the new coordinates of the simulation points
            new_xy = [rotate_and_translate(
                p[0], p[1], alpha_step[ref_id[i]], ref_xy[ref_id[i]][0], ref_xy[ref_id[i]][1])
                      for i, p in enumerate(points_xy)]
            # if the x' is positive then change the assigned reference point to the next
            ref_id = [ref_id[i]+1 if p[0] > 0 else ref_id[i] for i, p in enumerate(new_xy)]
            ref_id = [x if x < len(ref_xy) else 0 for x in ref_id]
            # compute the reward for each point from 0 to end-1
            reward = np.array([step_count(ref_id[i], ref_id[i+1], len(ref_xy)) for i in range(len(new_xy)-1)])
            if reward_function == 'spatial_progress':
                reward=step_dist*reward
            
        reward_df = pd.DataFrame({'r': reward})

        # Add next state
        state_prime = {}
        for s, sp in zip(state_cols, state_prime_cols):
            state_prime[sp] = np.array(lap_df[s])[1:]

        state_prime_df = pd.DataFrame(state_prime)

        # Add absorbing column

        absorbing = np.full([n_samples-1,], False)
        # If the lap is partial do not mark the last sample as the absorbing
        # otherwise do it
        if ~(lap_df.is_partial.values[0]):
            absorbing[-1] = True
        #absorbing[-1] = True
        absorbing_df = pd.DataFrame({'absorbing': absorbing})

        episodes[e] = pd.concat((nlap_df, timestamp_df, state_df, action_df, reward_df, state_prime_df, absorbing_df), axis=1)
    
    if reward_function == 'progress':
        # remove the reward of the reference to have no offset
        # reference_reward = episodes[df[df['isReference']]['NLap'].values[0]]['r'].values[0]
        for e in np.unique(df['NLap']):
            episodes[e]['r'] = episodes[e]['r'] - reference_reward

    # concatenate all the episodes to have the full dataset
    dataset = pd.concat(episodes)

    if reward_function == 'progress':
        # In case of progress reward it is possible that some samples have too high or too low values due to out-track
        # position. Thus we remove all the samples with reward higher than +5 and lower than -20
        clip_mask = (np.array(dataset['r']) >= -20) & (np.array(dataset['r']) <= 5)
        dataset = dataset[clip_mask]

    return dataset


def step_count(step_i, step_next, max_n):
    count = 0
    si = step_i
    sn = step_next
    
    stop = False
    
    while not stop:
        
        if si == sn:
            stop = True
            
        else:
            count += 1
            
            if si == max_n:
                si = 0
            else:
                si += 1
    
    return count


def create_kdt_ad(dataset, s_norm, filter_outliers, n_jobs, action_dispatcher, ad_param):
    
    # Filter the state features to consider for the knn
    state_mask = [i for i, s in enumerate(state_cols) if s in knn_state_cols]

    # create kdt using knn_state_cols state variables
    state_variables = [state_cols[i] for i in state_mask]

    data = list(dataset[state_variables].values)
    actions = list(map(lambda x: list(x), dataset[action_cols].values))
                   
    if s_norm:
        s_scaler = MinMaxScaler()
        data = s_scaler.fit_transform(data)
    else:
        s_scaler=None
    
    if filter_outliers:
        a_scaler = MinMaxScaler().fit(actions)
    else:
        a_scaler=None
    
    state_kdt = KDTree(data)
    
    ad = action_dispatcher(actions, state_mask, state_kdt, s_scaler, a_scaler, filter_outliers, n_jobs, ad_param)
    return ad


def create_fixed_kdt_ad(dataset, norm, k=100):

    # Filter the state features to consider for the knn
    state_mask = [i for i, s in enumerate(state_cols) if s in knn_state_cols]

    # create kdt using knn_state_cols state variables
    state_variables = [state_cols[i] for i in state_mask]

    data = list(dataset[state_variables].values)
    
    if norm:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        scaler=False

    state_kdt = KDTree(data)

    ad = FixedKDTActionDispatcher(list(map(lambda x: list(x), dataset[action_cols].values)), state_mask, state_kdt,
                                  scaler, k)
    return ad


def create_radial_kdt_ad(dataset, norm, r):
    # For each sample find the states with a distance lower or equal than r and take their actions

    # Filter the state features to consider for the knn
    state_mask = [i for i, s in enumerate(state_cols) if s in knn_state_cols]

    # create kdt using knn_state_cols state variables
    state_variables = [state_cols[i] for i in state_mask]

    data = list(dataset[state_variables].values)

    if norm:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    else:
        scaler=False

    state_kdt = KDTree(data)

    ad = RadialKDTActionDispatcher(list(map(lambda x: list(x), dataset[action_cols].values)), state_mask, state_kdt,
                                   scaler, r)
    return ad


def create_action_combinations(dataset, n_throttle=0, n_brake=0, n_steer=0, filter_actions=False):
    """Given a dataset, i.e., a sequences of (s,a,r,s') samples, it returns
    the list of all possible action combinations. If additional parameters
    are provided then it returns the combinations with subsampled actions.

    Input:
    dataset -- DataFrame with s,a,r,s' samples

    Keyword arguments:
    n_throttle -- number of elements for rThrottlePedal subsampling
    n_brake -- number of elements for pBrakeF subsampling
    n_steer -- number of elements for aSteerWheel subsampling
    filter_actions -- if True remove unfeasible action combinations
    
    Output:
    actions -- action combinations
    sub_actions -- combinations of subsampled actions
    """
    
    # get the unique values for each action    
    a_values = []
    for i,a in enumerate(action_cols):
        a_values.append(np.unique(dataset[a]))
    
    # create all the combinations
    actions = list(itertools.product(*a_values))
    
    # remove unfeasible actions:
    #    - up, down shift at the same time
    #    - throttle, brake at the same time
    #    - throttle and down shift
    #    - brake and up shift
    
    if filter_actions:
        
        wrong_shift = np.array(list(map(lambda x: x[3] == 1 and x[4] == 1, actions)))
        wrong_thr_brk = np.array(list(map(lambda x: x[0] > 0.0 and x[2] > 0.0, actions)))
        wrong_thr_down = np.array(list(map(lambda x: x[2] > 0.0 and x[3] == 1, actions)))
        wrong_brk_up = np.array(list(map(lambda x: x[0] > 0.0 and x[1] == 1, actions)))
        
        wrong_mask = wrong_shift | wrong_thr_down | wrong_brk_up | wrong_thr_brk
        
        actions = [actions[i] for i in range(len(actions)) if not wrong_mask[i]]
    
    # if no number of subactions is provided then return all the actions
    if n_throttle == 0:
        return actions
    
    # subsample actions
    sub_a_values = []
    for i,a in enumerate(action_cols):
        
        if a == 'rThrottlePedal':
            step = int(len(a_values[i]) / n_throttle)
            
        elif a == 'pBrakeF':
            step = int(len(a_values[i]) / n_brake)
            
        elif a == 'aSteerWheel':
            step = int(len(a_values[i]) / n_steer)

        else:
            step = 1
        
        sub_a_values.append(a_values[i][::step])
    
    # create all the combinations with sub_actions      
    sub_actions = list(itertools.product(*sub_a_values))
    
    if filter_actions:
        
        wrong_shift = np.array(list(map(lambda x: x[3] == 1 and x[4] == 1, sub_actions)))
        wrong_thr_brk = np.array(list(map(lambda x: x[0] > 0.0 and x[2] > 0.0, sub_actions)))
        wrong_thr_down = np.array(list(map(lambda x: x[2] > 0.0 and x[3] == 1, sub_actions)))
        wrong_brk_up = np.array(list(map(lambda x: x[0] > 0.0 and x[1] == 1, sub_actions)))
        
        wrong_mask = wrong_shift | wrong_thr_down | wrong_brk_up | wrong_thr_brk
        
        sub_actions = [sub_actions[i] for i in range(len(sub_actions)) if not wrong_mask[i]]
    
    return actions, sub_actions
