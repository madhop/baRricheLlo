import numpy as np
import csv
import os


def extract_features(s, env):
    scaled = env.scale_state
    assert len(s) == env.dim

    dist_junct_ego, ego_speed, ego_acceleration, entered_in_junct, \
        dist_veh1_road1, speed_veh1_road1, dist_veh2_road1, speed_veh2_road1,\
        dist_veh1_road2, speed_veh1_road2, dist_veh2_road2, speed_veh2_road2, \
        dist_veh1_road3, speed_veh1_road3, dist_veh2_road3, speed_veh2_road3,\
        dist_veh1_road4, speed_veh1_road4, dist_veh2_road4, speed_veh2_road4 = s[:-8]
    occ_flags = s[-8:]

    scale_factor_distance = (env.max_distance if scaled else 1)
    scale_factor_speed = (env.target_speed if scaled else 1)

    dist_junct_ego = dist_junct_ego*scale_factor_distance
    ego_speed = ego_speed*scale_factor_speed
    dist_veh1_road1 = dist_veh1_road1 * scale_factor_distance
    speed_veh1_road1 = speed_veh1_road1 * scale_factor_speed
    dist_veh1_road2 = dist_veh1_road2 * scale_factor_distance
    speed_veh1_road2 = speed_veh1_road2 * scale_factor_speed
    dist_veh2_road1 = dist_veh2_road1 * scale_factor_distance
    speed_veh2_road1 = speed_veh2_road1 * scale_factor_speed
    dist_veh2_road2 = dist_veh2_road2 * scale_factor_distance
    speed_veh2_road2 = speed_veh2_road2 * scale_factor_speed
    dist_veh1_road3 = dist_veh1_road3 * scale_factor_distance
    speed_veh1_road3 = speed_veh1_road3 * scale_factor_speed
    dist_veh1_road4 = dist_veh1_road4 * scale_factor_distance
    speed_veh1_road4 = speed_veh1_road4 * scale_factor_speed
    dist_veh2_road3 = dist_veh2_road3 * scale_factor_distance
    speed_veh2_road3 = speed_veh2_road3 * scale_factor_speed
    dist_veh2_road4 = dist_veh2_road4 * scale_factor_distance
    speed_veh2_road4 = speed_veh2_road4 * scale_factor_speed

    list_speed = list([speed_veh1_road1, speed_veh2_road1, speed_veh1_road2, speed_veh2_road2,
                       speed_veh1_road3, speed_veh2_road3, speed_veh1_road4, speed_veh2_road4])
    list_dist = list([dist_veh1_road1, dist_veh2_road1, dist_veh1_road2, dist_veh2_road2,
                      dist_veh1_road3, dist_veh2_road3, dist_veh1_road4, dist_veh2_road4])

    dt, v0, a0 = dist_junct_ego, ego_speed, ego_acceleration
    # t_max_speed = (scale_factor_speed - v0) / env.MAX_ACEL
    # d_max_speed = v0 * t_max_speed + 0.5 * env.MAX_ACEL * t_max_speed ** 2
    # if d_max_speed > dt:
    #     # solve_second_order
    #     a = 0.5 * env.MAX_ACEL
    #     b = v0
    #     c = - dt
    #     x_1 = (-b + (b ** 2 - 4 * a * c) ** .5) / (2 * a)
    #     x_2 = (-b - (b ** 2 - 4 * a * c) ** .5) / (2 * a)
    #     tte_ego = max(x_1, x_2)
    # else:
    #     t_2 = (dt - d_max_speed) / scale_factor_speed
    #     tte_ego = (t_2 + t_max_speed)
    speed_min_ego = 1.
    if ego_speed < 1.:
        tte_ego = dist_junct_ego / 1.
    else:
        tte_ego = dist_junct_ego/(ego_speed + 1e-20)

    speed_min_oth = 0.1
    tte_speed = list()
    for ind, el in enumerate(list_speed):
        if el < speed_min_oth:
            tte = list_dist[ind] / speed_min_oth
            list_speed[ind] = el = speed_min_oth
        else:
            tte = list_dist[ind] / (el + 1e-20)
        tte_speed.append((tte, el))

    #tte_speed.sort(key=lambda x: x[0])

    feature_list = list()
    const_acc = 6.
    tau = 1.
    for ind, el in enumerate(tte_speed):
        future_dist = (el[0] - tte_ego) * el[1]
        feature = el[1] ** 2 + const_acc * (tau * el[1] - future_dist)
        feature_list.append(feature)

    return tte_ego, ego_speed, ego_acceleration, dist_junct_ego, entered_in_junct, feature_list, occ_flags


class RuleBasedPolicy:
    NUM_THETA = 8
    rho_att = ['tte_critic_th_mean', 'tte_critic_th_var', 'tte_crash_th_mean', 'tte_crash_th_var',
               'tte_ego_critic_th_1_mean', 'tte_ego_critic_th_1_var', 'tte_ego_critic_th_2_mean',
               'tte_ego_critic_th_2_var',
               'tte_ego_critic_th_3_mean', 'tte_ego_critic_th_3_var', 'tte_ego_critic_th_4_mean',
               'tte_ego_critic_th_4_var',
               'tte_ego_critic_th_5_mean', 'tte_ego_critic_th_5_var', 'ego_close_th_mean', 'ego_close_th_var',
               'tte_ss_th_mean', 'tte_ss_th_var']
    # tte_critic_th, tte_crash_th, tte_ego_critic_th, ego_close_th

    default_values = np.array(
        [[300., 3.],
         [-20, 1.],
         [4, 1.5],
         [4, 1.5],
         [4, 1.5],
         [4, 1.5],
         [4, 1.5],
         [7, 1.5],
         [7, 1.5]]

    )

    def __init__(self, init_values, rho_att):
        assert len(rho_att) == init_values.shape[0] * init_values.shape[1]
        if init_values is None:
            init_values = RuleBasedPolicy.default_values
        self.epsilon = 1e-24
        self.rho = np.array(init_values, copy=True)
        self.theta = np.zeros(init_values.shape[0])
        self.t = 0
        self.rho_att = rho_att
        self.resample()

    def resample(self):
        for i, th in enumerate(self.theta):
            self.theta[i] = np.random.normal(self.rho[i, 0], np.exp(self.rho[i, 1]))
        return np.copy(self.theta)

    def act(self, s):
        pass



    def eval_params(self):
        return np.copy(self.rho)

    def get_theta(self):
        return np.copy(self.theta[:])

    def set_params(self, rho):
        self.rho[:, :] = rho[:, :]
        self.resample()

    def set_theta(self, theta):
        self.theta[:] = theta[:]

    def set_actor_params(self, theta):
        self.set_theta(theta)

    def set_mean(self):
        self.theta[:] = self.rho[:, 0]

    def eval_gradient(self, thetas, returns, use_baseline=True):

        means = self.rho[:, 0]
        sigmas = np.exp(self.rho[:, 1])
        n = len(means)
        b = 0
        gradients = []
        gradient_norms = []
        self.t += 1
        for i, theta in enumerate(thetas):
            d_mu = (theta - means) / (sigmas**2 + self.epsilon)
            d_sigma = ((theta - means) ** 2 - sigmas ** 2) / (sigmas ** 3 + self.epsilon)
            gradients.append(np.concatenate((d_mu, d_sigma)))
            gradient_norms.append(np.linalg.norm(np.concatenate((d_mu, d_sigma))))
        if use_baseline:
            gradient_norms = np.array(gradient_norms)
            num = (returns * gradient_norms ** 2).mean()
            den = (gradient_norms ** 2).mean()
            b = num / den
        gradient = (gradients * (np.array(returns) - b)[:, np.newaxis]).mean(axis=0)
        d_mu = gradient[:n]
        d_sigma = gradient[n:]
        return np.stack([d_mu, d_sigma]).T

    def eval_natural_gradient(self, thetas, returns, use_baseline=True):

        means = self.rho[:, 0]
        sigmas = np.exp(self.rho[:, 1])
        n = len(means)
        b = 0
        gradients = []
        gradient_norms = []
        self.t += 1
        for i, theta in enumerate(thetas):
            d_mu = (theta - means)
            d_sigma = ((theta - means)**2 - sigmas**2) / (2 * sigmas**2 + self.epsilon)
            
            gradients.append(np.concatenate((d_mu, d_sigma)))
            gradient_norms.append(np.linalg.norm(np.concatenate((d_mu, d_sigma))))
        if use_baseline:
            gradient_norms = np.array(gradient_norms)
            num = (returns * gradient_norms ** 2).mean()
            den = (gradient_norms**2).mean()
            b = num / den
        gradient = (gradients * (np.array(returns) - b)[:, np.newaxis]).mean(axis=0)

        d_mu = gradient[:n]
        d_sigma = gradient[n:]

        return np.stack([d_mu, d_sigma]).T


    def show_theta(self):
        print(self.theta)

    @property
    def std(self):
        return np.exp(self.rho[:, 1])

    @property
    def mean(self):
        return self.rho[:, 0]

    def save(self, file_path):

        if not os.path.isfile(file_path):
            with open(file_path, 'w', newline='') as out_file:
                dicti_writer = csv.DictWriter(out_file, fieldnames=self.get_att_names())
                dicti_writer.writeheader()

        rho_to_save = self.eval_params()
        with open(file_path, 'a') as out_file:
            dicti_writer = csv.DictWriter(out_file, fieldnames=self.get_att_names())
            row = {}
            for i, attr in enumerate(self.get_att_names()):
                j, k = i // 2, i % 2
                row[attr] = rho_to_save[j][k]
            dicti_writer.writerow(row)

    def get_att_names(self):
        return self.rho_att

    def renyi(self, other, alpha=2.):
        tol = 1e-45
        assert isinstance(other, RuleBasedPolicy)
        var_alpha = alpha * other.std**2 + (1. - alpha) * self.std**2
        return alpha/2. * np.sum((self.mean - other.mean)**2 / (var_alpha + tol), axis=-1) - \
               1./(2*(alpha - 1)) * (np.log(np.prod(var_alpha, axis=-1) + tol) -
                   np.log(np.prod(self.std**2, axis=-1) + tol) * (1-alpha)
                                - np.log(np.prod(other.std ** 2, axis=-1) + tol) * alpha)

    def eval_renyi(self, other, order=2):

        return self.renyi(other, alpha=order)

    def eval_fisher(self):
        mean_fisher_diag = np.exp(-2 * self.rho[:, 1])
        cov_fisher_diag = mean_fisher_diag * 0 + 2
        fisher_diag = np.concat([mean_fisher_diag, cov_fisher_diag], axis=0)
        return np.ravel(fisher_diag)

    def fisher_product(self, x):

        return x / self.eval_fisher()

    @staticmethod
    def get_ttc(follower_speed, leader_speed, leader_distance):
        diff_speed = follower_speed - leader_speed
        if diff_speed > 0:
            return leader_distance / diff_speed
        else:
            return np.inf


