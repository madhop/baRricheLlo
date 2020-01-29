from run_torcs_FQI import playGame

algorithm_name = 'temporal_penalty_xy_reward_model_old.pkl'#'temporal_penalty_xy_reward_model.pkl'#'temporal_penalty_xy_reward_boltzmann_model.pkl'#'temporal_penalty_reward_model.pkl'#'temporal_penalty_reward_greddy_model.pkl'
policy_type = 'boltzmann'#'greedy'#'greedy_noise'#
episode_count = 5
playGame(algorithm_name, policy_type, episode_count)
file_name = "preprocessed_torcs_algo"
output_file = "trajectory/dataset_offroad.csv"
preprocess_raw_torcs(file_name, output_file)
buildDataset(raw_input_file_name = file_name, output_file = output_file, header = False)
