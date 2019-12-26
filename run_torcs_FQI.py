from gym_torcs import TorcsEnv
from agent_FQI import AgentFQI, AgentMEAN
import pandas as pd
import numpy as np
import compute_state_features as csf
from fqi.utils import *
from fqi.reward_function import *
import time
import os
from utils_torcs import *
#from datetime import date

def appendObs(store_obs, ob, action):
    for k in torcs_features:
        store_obs[k] = np.append(store_obs[k], ob[k])
    for a_idx, a in enumerate(torcs_actions):
        store_obs[a] = np.append(store_obs[a], action[a_idx])

def emptyStoreObs():
    store_obs = { k : [] for k in torcs_features}
    for a in torcs_actions:
        store_obs[a] = []
    return store_obs

def playGame():
    start_line = False
    track_length = 5783.85
    #raw_output_name = 'raw_torcs_data/raw_data_algo_' + str(date.today().year) + '_' +  str(date.today().month) + '_' + str(date.today().day) + '.csv'
    raw_output_path = 'raw_torcs_data/'
    raw_output_name = 'raw_data_algo.csv'
    algorithm_name = 'model_r_speed_50laps_pc.pkl'#'first_model.pkl'
    policy_path = 'model_file/policy_' + algorithm_name
    action_dispatcher_path = 'model_file/AD_' + algorithm_name
    episode_count = 2
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    ref_df = pd.read_csv('trajectory/ref_traj.csv') # reference trajectory
    ref_df.columns = ref_traj_cols
    store_obs = { k : [] for k in torcs_features}
    for a in torcs_actions:
        store_obs[a] = []

    reward_function = reward_function = Speed_projection(ref_df)

    agent = AgentFQI(ref_df, policy_path, action_dispatcher_path)
    #agent = AgentMEAN()

    # Generate a Torcs environment
    env = TorcsEnv(reward_function, vision=False, throttle=True, gear_change=True, brake=True) #gear_change = False -> automatic gear change

    print("TORCS Experiment Start.")
    for i in range(episode_count):
        print("Episode : " + str(i))

        if np.mod(i, 3) == 0:
            # Sometimes you need to relaunch TORCS because of the memory leak error
            ob = env.reset(relaunch=True)
            ob_2 = ob_1 = ob
        else:
            ob = env.reset()
            ob_2 = ob_1 = ob

        # at the moment we need to slow time, because the agent is too slow
        os.system('sh slow_time_down.sh')
        time.sleep(0.5)

        total_reward = 0.
        for j in range(max_steps):
            if ob['distFromStart'] < 100 and not start_line:    # just passed start line
                print('----',j)
                start_line = True
                action = [0,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action)
            elif ob['distFromStart'] < 5615.26 and not start_line:   # exit from pit stop
                print('-', j)
                action = [0.02,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action)
            elif ob['distFromStart'] < 5703.24 and not start_line:
                print('--', j)
                action = [-0.028,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action)
            elif ob['distFromStart'] < track_length and not start_line:
                print('---', j)
                action = [0,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action)
            else:
                action = agent.act(ob, ob_1, ob_2, action, reward)   #AgentFQI
                print('Action:', action)
                # if no action was returned
                if len(action) == 0:
                    print('no actions')
                    start_line = False
                    break
                #action = agent.act(ob)    #AgentMEAN
                ob_2 = ob_1
                ob_1 = ob
                # store observation and action, to be saved in CSV
                appendObs(store_obs, ob, action)

                ob, reward, done, _ = env.step(action)
                total_reward += reward

                step += 1
                if done:
                    appendObs(store_obs, ob, action)
                    start_line = False
                    break

        print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    print("Saving raw data to CSV")
    df = pd.DataFrame(store_obs)
    df.to_csv(index = False, path_or_buf = raw_output_path + raw_output_name, mode = 'w', header = True)
    df.to_csv(index = False, path_or_buf = raw_output_path + 'all_' + raw_output_name, mode = 'a', header = True)
    env.end()  # This is for shutting down TORCS
    print('raw data output:', raw_output_path + raw_output_name)
    print("Finish.")


if __name__ == "__main__":
    playGame()
