import sys
import os
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '..'))
import pickle
import argparse
from fqi.dataset_preprocessing import *
from trlib.policies.double_qfunction import DoubleQFunction


def run_evaluation(algorithm_path, track_file_name, data_path, n_jobs, output_path, output_name, filter_actions,
                   action_dispatcher_path=''):
    
    # load algorithm object
    with open(algorithm_path, 'rb') as alg:
        algorithm = pickle.load(alg)   
    Q = algorithm._policy.Q
    
    # if we are in the double fqi framework
    double_fqi = isinstance(Q, DoubleQFunction)
    
    if double_fqi:
        if hasattr(Q._regressor[0], 'n_jobs'):
            for i in range(len(Q._regressor)):
                Q._regressor[i].n_jobs = n_jobs
    else:
        if hasattr(Q._regressor, 'n_jobs'):
            Q._regressor.n_jobs = n_jobs
        
    # load track file
    track = pd.read_csv(os.path.join(data_path, track_file_name+'.csv'), dtype={'isReference': bool})

    # get list of actions
    if len(action_dispatcher_path) == 0:
        # get the list of the possible action combinations
        actions = create_action_combinations(track, filter_actions=filter_actions)

    else:
        # Find the nearest actions for each state in the track data frame

        # get the list of all the states in the simulation
        all_states = track[state_cols].values

        with open(action_dispatcher_path, 'rb') as ad:
            action_dispatcher = pickle.load(ad)

        actions = {}
        action_list = action_dispatcher.get_actions(all_states)
        states_to_remove = []
        for i in range(len(all_states)):
            
            # if the action set of the state is empty save i to remove it
            if len(action_list[i]) == 0:
                states_to_remove.append(all_states[i, :])
            else:
                actions[tuple(all_states[i, :])] = action_list[i]
            
        # Filter out the states that do not have actions in the action set because of they are too far from
        # the states in the training set
        empty_mask = np.isin(track[state_cols].values, states_to_remove).all(axis=1)
        if np.count_nonzero(empty_mask) > 0:
            #track = track[~empty_mask]
            print('Detected {} empty states'.format(np.count_nonzero(empty_mask)))
        
    # dictionary containing for each lap the q-values of pilot and policy
    q_values = {}
    laps = np.unique(track['NLap'])
    for l in laps:
        
        print('Processing {} of {}'.format(l, len(laps)))
        lap_df = track[track['NLap'] == l]
        n_samples = lap_df.shape[0]
        lap_empty_mask = empty_mask[track.NLap == l]
        
        pilotQ = np.full((n_samples,), np.nan)
        policyQ = np.full((n_samples,), np.nan)
        policyActions = np.full((n_samples, len(action_cols)), np.nan)
         
        # Compute the Q value of the pilot actions
        sa = np.array(pd.concat([lap_df[state_cols], lap_df[action_cols]], axis=1))
        
        pilotQ[~lap_empty_mask] = Q.values(sa)
        print('Computed pilot Q values')

        # Compute the Q value of the actions of the learned policy
        print('Computing policy Q values')
        lap_states = lap_df[state_cols].values
        policyQ[~lap_empty_mask], policyActions[~lap_empty_mask, :] = Q.max(lap_states, actions)

        q_values[l] = [pilotQ, policyQ, policyActions]
    
    print('Saving results')
 
    with open(output_path + '/'+output_name+'.pkl', 'wb') as output:
        pickle.dump(q_values, output, pickle.HIGHEST_PROTOCOL)
    
    print('Saved evaluation')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--algorithm_path", type=str)
    parser.add_argument("--track_file_name", type=str , help='Name of the data file containing the laps')
    parser.add_argument("--data_path", type=str,
                        default=os.path.join(file_path, '..', '..', '..', '..', '..', '..', '..', 'data',
                                             'ferrari', 'driver', 'datasets', 'csv'),
                        help='Path of the folder containig csv data files')
    parser.add_argument("--n_jobs", type=int, default=1)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--output_name", type=str, default='evaluation')
    parser.add_argument("--filter_actions", type=bool, default=False)
    parser.add_argument("--action_dispatcher", type=str, default='', help='path of the object ActionDispatcher')

    args = parser.parse_args()
    out_dir = args.output_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    run_evaluation(args.algorithm_path, args.track_file_name, args.data_path, args.n_jobs, args.output_path,
                   args.output_name, args.filter_actions, args.action_dispatcher)
