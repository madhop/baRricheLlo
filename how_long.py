from gym_torcs import TorcsEnv
from agent_FQI import AgentFQI, AgentMEAN
import pandas as pd
import numpy as np
import compute_state_features as csf
from fqi.utils import *
import time


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

agent = AgentFQI(ref_df, policy_path, action_dispatcher_path)

ref_df = pd.read_csv('trajectory/ref_traj.csv') # reference trajectory
ref_df.columns = ref_traj_cols

ob_2 = {'angle': (0.0552266), 'curLapTime': (-0.582), 'distFromStart': (5496.38), 'Acceleration_x': (-47.8569), 'Acceleration_y': (-0.286503), 'Gear': (0.), 'rpm': (9367.96), 'speed_x': (296.36), 'speed_y': (0.101232), 'speed_z': (0.225744), 'x': (340.77), 'y': (1148.21), 'z': (0.228214), 'roll': (-6.89871e-05), 'pitch': (0.00496075), 'yaw': (-0.00677342), 'speedGlobalX': (82.3205), 'speedGlobalY': (-0.52988)}

ob_1 = {'angle': (0.0552266), 'curLapTime': (-0.562), 'distFromStart': (5496.38), 'Acceleration_x': (-47.8569), 'Acceleration_y': (-0.286503), 'Gear': (0.), 'rpm': (9648.33), 'speed_x': (296.36), 'speed_y': (0.101232), 'speed_z': (0.225744), 'x': (340.77), 'y': (1148.21), 'z': (0.228214), 'roll': (-6.89871e-05), 'pitch': (0.00496075), 'yaw': (-0.00677342), 'speedGlobalX': (82.3205), 'speedGlobalY': (-0.52988)}

ob = {'angle': (0.0552266), 'curLapTime': (-0.542), 'distFromStart': (5496.38), 'Acceleration_x': (-47.8569), 'Acceleration_y': (-0.286503), 'Gear': (0.), 'rpm': (9928.97), 'speed_x': (296.36), 'speed_y': (0.101232), 'speed_z': (0.225744), 'x': (340.77), 'y': (1148.21), 'z': (0.228214), 'roll': (-6.89871e-05), 'pitch': (0.00496075), 'yaw': (-0.00677342), 'speedGlobalX': (82.3205), 'speedGlobalY': (-0.52988)}

prev_action = [-0.004653,  0.,        1.      ]

n = 10
start = time.time()
for _ in range(n):
    action, end_of_lap = agent.act(ob, ob_1, ob_2, prev_action, reward)   #AgentFQI

elapsed_time = time.time()-start
print('it took:', elapsed_time)
print('avg:', elapsed_time/n)
