from gym_torcs import TorcsEnv
from human_agent import HumanAgent
import numpy as np
import pandas as pd
from state import *

def emptyObsDic():
    # return dictionary of the observation (to save csv file)
    d = {}
    for s in state_cols:
        d[s] = []
    return d
    """return {'angle': [], 'curLapTime': [], 'distFromStart': [],
            #'focus': [],
            'gear': [], 'lastLapTime': [], 'rpm': [], 'speed_x': [], 'speed_y': [], 'speed_z':[],
            #'track': [],
            'trackPos':[],
            #'wheelSpinVel': [],
            'x': [], 'y':[], 'z':[], 'roll':[], 'pitch': [], 'yaw':[], 'speedGlobalX':[], 'speedGlobalY':[],
            'steering': [], 'acceleration': [], 'gear_rec': [], 'brake': []}"""

episode_count = 1
max_steps = 10000
reward = 0
done = False
#collect_data_mode = False
collect_data_mode = True
step = 0

obs_dic = emptyObsDic()
#df = pd.DataFrame(obs_dic)
#df.to_csv(index = False, path_or_buf = 'ref_trajectory.csv')

# check if csv with reference trajectory exists
try:
    ref_trajectory_df = pd.read_csv('ref_trajectory_monza.csv')  # this is the trajectory of the best lap (until now)
    bestTime = ref_trajectory_df.iloc[-1,4]    # the best time is on the last row of the best trajectory ('lastLapTime')
    track_length = ref_trajectory_df.iloc[-2,2]-20 #5784.10  Forza
except:
    bestTime = np.inf
    track_length = 500#5700
print('bestTime: ', bestTime)
save = False

#if csv with car trajectory does not exist, build empty one with header
"""try:
    car_trajectory_df = pd.read_csv('car_trajectory_monza.csv')
except:
    df = pd.DataFrame(obs_dic)
    df.to_csv(index = False, path_or_buf = 'car_trajectory_monza.csv')"""

# Generate a Torcs environment
env = TorcsEnv(vision=False, throttle=True, gear_change=False, brake=True)
agent = HumanAgent(max_steps, use_logitech_g27=False)
#agent = HumanAgent(max_steps, use_logitech_g27=True)


print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    total_reward = 0.

    for j in range(max_steps):
        action = agent.act(ob, reward, done, step)  #Steering/Acceleration/Gear/Brake

        ob, reward, done, _ = env.step(action)
        #angle, curLapTime, distFromStart, focus, gear, lastLapTime, rpm, speed_x, speed_y, speed_z, track, trackPos, wheelSpinVel, x, y, z, roll, pitch, yaw, speedGlobalX, speedGlobalY = ob
        #angle, tgAngle, segAngle, curLapTime, distFromStart, lastLapTime, speed_x, speed_y, speed_z, trackPos, x, y, yaw = ob
        for s in state_cols:
            obs_dic[s] = np.append(obs_dic[s], ob[s])
        if ob['distFromStart'] > track_length and save == False:    # get ready only at the end of the lap
            print("Almost at the end of the lap")
            save = True
        if ob['distFromStart'] < 3:   # as soon as the car pass the finish line check if it was the best time and save new trajectory
            print('Lap Finished')
            if save == True and ob['lastLapTime'] > 0:    # this check is to save just once at each lap
                save = False    # so that it save only once
                df = pd.DataFrame(obs_dic)
                """if lastLapTime < bestTime:  # if it is the best lap time, save as reference trajectory
                    bestTime = lastLapTime
                    print("NEW BEST LAP TIME: ", lastLapTime)
                    print('Save to CSV')
                    df.to_csv(index = False, path_or_buf = 'ref_trajectory_monza.csv')"""
                df.to_csv(index = False, path_or_buf = 'track_data/track_data_monza.csv', mode = 'w', header = True)
                #df.to_csv(index = False, path_or_buf = 'track_data/track_data_monza.csv', mode = 'a', header = False)

            obs_dic = emptyObsDic()

        #print("gym_obs: ", ob)

        total_reward += reward

        step += 1
        if done:
            break

    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

agent.end(step == max_steps)
env.end()  # This is for shutting down TORCS
print("Finish.")
