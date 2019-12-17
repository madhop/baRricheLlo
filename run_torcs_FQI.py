from gym_torcs import TorcsEnv
from agent_FQI import AgentFQI, AgentMEAN
import pandas as pd
import numpy as np
import compute_state_features as csf
from fqi.utils import *
from fqi.reward_function import *
import time
import os


def playGame():
    start_line = False
    track_length = 5783.85
    algorithm_name = 'model_r_speed_50laps_pc.pkl'#'first_model.pkl'
    policy_path = 'model_file/policy_' + algorithm_name
    action_dispatcher_path = 'model_file/AD_' + algorithm_name
    vision = False
    episode_count = 10
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    ref_df = pd.read_csv('trajectory/ref_traj.csv') # reference trajectory
    ref_df.columns = ref_traj_cols

    reward_function = reward_function = Speed_projection(ref_df)

    agent = AgentFQI(ref_df, policy_path, action_dispatcher_path)
    #agent = AgentMEAN()

    # Generate a Torcs environment
    env = TorcsEnv(reward_function, vision=vision, throttle=True, gear_change=True, brake=True) #gear_change = False -> automatic gear change

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
            if ob['distFromStart'] < 100 and not start_line:
                print('---',j)
                start_line = True
                action = [0,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action, False)
            elif ob['distFromStart'] < 5615.26 and not start_line:   # at the beginning just throttle a bit
                print('-', j)
                action = [0.02,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action, False)
            elif ob['distFromStart'] < 5703.24 and not start_line:
                print('--', j)
                action = [-0.028,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action, False)
            elif ob['distFromStart'] < track_length and not start_line:
                print('--', j)
                action = [0,0,1, 7]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action, False)
            else:
                action, end_of_lap, done = agent.act(ob, ob_1, ob_2, action, reward)   #AgentFQI
                action = np.append(action, [0]) # add fake gear
                print('Action:', action)
                if ob['damage'] > ob_1['damage']:
                    done = True
                if done:
                    print('No actions')
                    start_line = False
                    break
                #action = agent.act(ob)    #AgentMEAN
                ob_2 = ob_1
                ob_1 = ob

                #ob, reward, done, _ = env.step(action, end_of_lap)
                ob, reward, done, _ = env.step(action, False)
                # check if hit the walls
                total_reward += reward

                step += 1
                done = False # TODO togli
                if done:
                    start_line = False
                    break

        print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame()
