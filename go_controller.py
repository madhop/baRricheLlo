import pickle
from fqi.reward_function import *
from fqi.utils import *
import pandas as pd
from controller import MeanController
from gym_torcs_ctrl import TorcsEnv


def starter(x):
    return [0, 0, 1, 7]

if __name__ == '__main__':
    #ref_df = pd.read_csv('../demonstrations/extracted_features/ref_traj.csv')
    ref_df = pd.read_csv('trajectory/ref_traj.csv')

    #demos_path = '../demonstrations/extracted_features/top_demonstrations.csv'
    demos_path = 'trajectory/dataset_offroad_human.csv'
    demos_penalty = pd.read_csv(demos_path)
    demos_penalty = demos_penalty[demos_penalty.time > 70]
    penalty = LikelihoodPenalty(kernel='gaussian', bandwidth=1.0)
    penalty.fit(demos_penalty[penalty_cols].values)
    reward_function = Temporal_projection(ref_df, penalty=None)

    env = TorcsEnv(reward_function, collision_penalty=-1000, state_cols=state_cols, ref_df=ref_df, vision=False,
                   throttle=True, gear_change=False, brake=True, start_env=False, damage_th=10.0, slow=False,
                   faster=False, graphic=True, starter=starter)

    C = MeanController(ref_df, env , gamma1=3, gamma2=73.5, gamma3=116, alpha1=0.5, alpha2=0.02, beta1=0.055, k=20)
    episode = C.playGame()
    C.env.end()
    pickle.dump(episode, open('../episode_to_check.pkl', 'wb'))
