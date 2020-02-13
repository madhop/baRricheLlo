from gym_torcs import TorcsEnv
from fqi.reward_function import *
from fqi.sars_creator import to_SARS
from fqi.utils import *
import pandas as pd
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from ddpg.ddpg_driver import DDPG
from stable_baselines.common.math_util import scale_action

# load reference trajectory
ref_df = pd.read_csv('trajectory/ref_traj.csv')
ref_df.columns = ref_traj_cols

simulations = pd.read_csv('trajectory/dataset_offroad_human.csv')
# Reward function
penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
right_laps = np.array([ 1.,  8.,  9., 11., 14., 16., 17., 20., 45., 46., 49.,  59., 62.])
penalty.fit(simulations[simulations.NLap.isin(right_laps)][penalty_cols].values)
reward_function = Temporal_projection(ref_df, penalty=penalty)

dataset = to_SARS(simulations, reward_function)

env = TorcsEnv(reward_function, state_cols=state_cols, ref_df=ref_df, vision=False, throttle=True,
               gear_change=False, brake=True, start_env=False, damage_th=0, slow=False, graphic=True)

model = DDPG.load("model_file/ddpg_600") #16 30 41 56 70 120 138
model.env = env


obs = env.reset()
reward_sum = 0.0
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_sum += reward
    #env.render()
    if done:
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

env.close()
