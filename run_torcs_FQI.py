from gym_torcs import TorcsEnv
from agent_FQI import AgentFQI, AgentMEAN
import pandas as pd
import numpy as np
import compute_state_features as csf

def playGame():
    vision = False
    episode_count = 10
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    ref_df = pd.read_csv('trajectory/ref_traj.csv') # reference trajectory
    ref_df.columns = ['curLapTime', 'Acceleration_x', 'Acceleration_y', 'speed_x', 'speed_y', 'x', 'y', 'alpha_step']

    # Generate a Torcs environment
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False, brake=True)
    agent = AgentFQI(ref_df)
    #agent = AgentMEAN()

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

        total_reward = 0.
        for j in range(max_steps):
            if j < 5:   # at the beginning just throttle
                action = [0,0.5,0, 0]
                ob_2 = ob_1
                ob_1 = ob
                ob, _, done, _ = env.step(action)
            else:
                action = agent.act(ob, ob_1, ob_2, reward, done)   #AgentFQI
                #action = agent.act(ob)    #AgentMEAN
                ob_2 = ob_1
                ob_1 = ob

                ob, reward, done, _ = env.step(action)
                #print(ob)
                total_reward += reward

                step += 1
                if done:
                    break

        print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame()
