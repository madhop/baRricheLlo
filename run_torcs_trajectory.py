from gym_torcs import TorcsEnv
from human_agent import HumanAgent
import numpy as np
import pandas as pd

def emptyObsDic():
    # return dictionary of the observation (to save csv file)
    return {'angle': [], 'curLapTime': [], 'distFromStart': [],
            #'focus': [],
            'gear': [], 'lastLapTime': [], 'rpm': [], 'speed_x': [], 'speed_y': [], 'speed_z':[],
            #'track': [],
            'trackPos':[],
            #'wheelSpinVel': [],
            'x': [], 'y':[], 'z':[], 'roll':[], 'pitch': [], 'yaw':[], 'speedGlobalX':[], 'speedGlobalY':[],
            'steering': [], 'acceleration': [], 'gear_rec': [], 'brake': []}

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
try:
    car_trajectory_df = pd.read_csv('car_trajectory_monza.csv')
except:
    df = pd.DataFrame(obs_dic)
    df.to_csv(index = False, path_or_buf = 'car_trajectory_monza.csv')

# Generate a Torcs environment
env = TorcsEnv(vision=False, throttle=True, gear_change=True, brake=True)
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

    if step == max_steps and not done and collect_data_mode:
        agent.next_dataset()

    for j in range(max_steps):
        action = agent.act(ob, reward, done, step)  #Steering/Acceleration/Gear/Brake

        ob, reward, done, _ = env.step(action)
        angle, curLapTime, distFromStart, focus, gear, lastLapTime, rpm, speed_x, speed_y, speed_z, track, trackPos, wheelSpinVel, x, y, z, roll, pitch, yaw, speedGlobalX, speedGlobalY = ob
        obs_dic['angle'] = np.append(obs_dic['angle'], angle)
        obs_dic['curLapTime'] = np.append(obs_dic['curLapTime'], curLapTime)
        obs_dic['distFromStart'] = np.append(obs_dic['distFromStart'], distFromStart)
        #obs_dic['focus'] = np.append(obs_dic['focus'], focus)
        obs_dic['gear'] = np.append(obs_dic['gear'], gear)
        obs_dic['lastLapTime'] = np.append(obs_dic['lastLapTime'], lastLapTime)
        obs_dic['rpm'] = np.append(obs_dic['rpm'], rpm)
        obs_dic['speed_x'] = np.append(obs_dic['speed_x'], speed_x)
        obs_dic['speed_y'] = np.append(obs_dic['speed_y'], speed_y)
        obs_dic['speed_z'] = np.append(obs_dic['speed_z'], speed_z)
        #obs_dic['track'] = np.append(obs_dic['track'], track)
        obs_dic['trackPos'] = np.append(obs_dic['trackPos'], trackPos)
        #obs_dic['wheelSpinVel'] = np.append(obs_dic['wheelSpinVel'], wheelSpinVel)
        obs_dic['x'] = np.append(obs_dic['x'], x)
        obs_dic['y'] = np.append(obs_dic['y'], y)
        obs_dic['z'] = np.append(obs_dic['z'], z)
        obs_dic['roll'] = np.append(obs_dic['roll'], roll)
        obs_dic['pitch'] = np.append(obs_dic['pitch'], pitch)
        obs_dic['yaw'] = np.append(obs_dic['yaw'], yaw)
        obs_dic['speedGlobalX'] = np.append(obs_dic['speedGlobalX'], speedGlobalX)
        obs_dic['speedGlobalY'] = np.append(obs_dic['speedGlobalY'], speedGlobalY)
        # Actions
        obs_dic['steering'] = np.append(obs_dic['steering'], action[0])
        obs_dic['acceleration'] = np.append(obs_dic['acceleration'], action[1])
        obs_dic['gear_rec'] = np.append(obs_dic['gear_rec'], action[2])
        obs_dic['brake'] = np.append(obs_dic['brake'], action[3])
        if distFromStart > track_length and save == False:    # get ready only at the end of the lap
            print("Almost at the end of the lap")
            save = True
        if distFromStart < 3:   # as soon as the car pass the finish line check if it was the best time and save new trajectory
            print('Lap Finisched')
            if save == True and lastLapTime > 0:    # this check is to save just once at each lap
                save = False    # so that it save only once
                df = pd.DataFrame(obs_dic)
                if lastLapTime < bestTime:  # if it is the best lap time, save as reference trajectory
                    bestTime = lastLapTime
                    print("NEW BEST LAP TIME: ", lastLapTime)
                    print('Save to CSV')
                    df.to_csv(index = False, path_or_buf = 'ref_trajectory_monza.csv')
                df.to_csv(index = False, path_or_buf = 'car_trajectory_monza.csv', mode = 'a', header = False)

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
