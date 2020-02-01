from run_torcs_FQI import playGame
from preprocess_raw_torcs_algo import *
from build_dataset_offroad import *

algorithm_name = 'temporal_penalty_xy_reward_model_old.pkl'#'temporal_penalty_xy_reward_model.pkl'#'temporal_penalty_xy_reward_boltzmann_model.pkl'#'temporal_penalty_reward_model.pkl'#'temporal_penalty_reward_greddy_model.pkl'
policy_type = 'boltzmann'#'greedy'#'greedy_noise'#
episode_count = 7
for i in range(4, 100, 5):
    policy_path = 'model_file/Policies/Policy' + str(i) + '.pkl'
    print('Policy' + str(i) + '.pkl')
    playGame(algorithm_name, policy_type, episode_count, policy_path)
    file_name = "preprocessed_torcs_algo"
    output_file = "trajectory/dataset_compare_Q.csv"
    preprocess_raw_torcs(file_name, output_file)
    if i == 4:
        buildDataset(raw_input_file_name = file_name, output_file = output_file, header = True)
    else:
        buildDataset(raw_input_file_name = file_name, output_file = output_file, header = False)
