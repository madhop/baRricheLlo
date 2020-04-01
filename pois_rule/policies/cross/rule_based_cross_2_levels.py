from env.cross_sumo_env import CrossSumoEnv
from policies.rule_based_policy import RuleBasedPolicy
import numpy as np


def act_closure(s, get_params):
    tte_ego, speed_ego, ego_acceleration, dist_junct_ego, entered_in_junct, feature_list = s[:]
    tte_critic_th, tte_crash_th, tte_ego_critic_th, ego_close_th = get_params()

    action_so_far = CrossSumoEnv.Actions.N.value  # the least constraining action

    return action_so_far



class RuleBasedPolicyCross2Levels(RuleBasedPolicy):
    NUM_THETA = 8
    # rho_att = ['tte_critic_th_mean', 'tte_critic_th_var', 'tte_crash_th_mean', 'tte_crash_th_var',
    #            'tte_ego_critic_th_mean', 'tte_ego_critic_th_var', 'ego_close_th_mean', 'ego_close_th_var']
    rho_att = ['tte_critic_th_mean', 'tte_critic_th_var', 'tte_crash_th_mean', 'tte_crash_th_var',
               'tte_ego_critic_th_mean', 'tte_ego_critic_th_var', 'ego_close_th_mean', 'ego_close_th_var',
               'tte_critic_th_2_mean', 'tte_critic_th_2_var', 'tte_crash_th_2_mean', 'tte_crash_th_2_var',
               'tte_ego_critic_th_2_mean', 'tte_ego_critic_th_2_var', 'ego_close_th_2_mean', 'ego_close_th_2_var']
    default_values = np.array(
        [[300., 3.],
         [-20, 2.],
         [4, 1.],
         [7, 1.],
         [300., 3.],
         [-20, 2.],
         [4, 1.],
         [7, 1.]
         ]
    )

    def __init__(self, init_values=None):

        if init_values is None:
            init_values = RuleBasedPolicyCross2Levels.default_values
        super(RuleBasedPolicyCross2Levels, self).__init__(init_values, rho_att=RuleBasedPolicyCross2Levels.rho_att)

    def act(self, s):

        tte_ego, speed_ego, dist_junct_ego, entered_in_junct, feature_list, occ_flags = s[:]
        tte_critic_th, tte_crash_th, tte_ego_critic_th, ego_close_th = self.theta[:]

        action_so_far = CrossSumoEnv.Actions.N.value  # the least constraining action

        return action_so_far




