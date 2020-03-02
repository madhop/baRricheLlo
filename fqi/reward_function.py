import numpy as np
from sklearn.neighbors import KDTree
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from fqi.utils import vectorized_rotate_and_translate, penalty_cols

# abstract
class Reward_function:
    def __init__(self, reference_trajectory, clip_range, reference_dt, sample_dt, penalty=None):  # dts in centiseconds
        self.ref_p = reference_trajectory[['xCarWorld', 'yCarWorld']].values
        self.ref_len = self.ref_p.shape[0]
        self.clipping = bool(clip_range)
        self.clip_range = clip_range
        self.ref_dt = reference_dt
        self.sample_dt = sample_dt
        self.penalty = penalty

    def _compute_reward(self, data):
        raise NotImplementedError

    def __call__(self, data):
        # the penalty term is computed considering the next state thus we pass the states from 1 to end
        return self._compute_reward(data) + (self.penalty.compute_offroad_penalty(data[penalty_cols].values[1:], data['trackPos'].values[1:]) if self.penalty else 0)
        #return self._compute_reward(data) + (self.penalty.compute_offroad_penalty(data[state_cols].values[1:], data['trackPos'].values[1:]) if self.penalty else 0)


class Discrete_temporal_reward(Reward_function):
    def __init__(self, ref_t, clip_range=(-20, 5), ref_dt=1, sample_dt=10, penalty=None):
        super().__init__(ref_t, clip_range, ref_dt, sample_dt, penalty)
        self.alpha_step = ref_t['alpha_step'].values
        self.kdtree = KDTree(self.ref_p)

    def _compute_reward(self, data):
        state_p = data[['xCarWorld', 'yCarWorld']].values
        _, ref_id = self.kdtree.query(state_p)
        ref_id = ref_id.squeeze()
        rot_p = vectorized_rotate_and_translate(state_p, self.alpha_step[ref_id], self.ref_p[ref_id])
        ref_id[rot_p[:, 0] > 0] += 1
        ref_id[ref_id == self.ref_len] = 0
        ref_time = ref_id[1:] - ref_id[:-1]
        ref_time[ref_time < -self.ref_len / 2] += self.ref_len
        reward = ref_time * self.ref_dt - self.sample_dt
        return reward


'''
Each state is represented by its continuous projection on the reference trajectory.
In particular each state is represented by the id of the preceding reference point (ref_id) and
the fraction of the segment starting at that point between the ref point and the projection (delta)
'''
# abstract
class Projection_reward(Reward_function):
    def __init__(self, ref_t, clip_range, ref_dt, sample_dt, penalty=None):
        super().__init__(ref_t, clip_range, ref_dt, sample_dt, penalty)
        self.kdtree = KDTree(self.ref_p)
        self.seg = np.roll(self.ref_p, -1, axis=0) - self.ref_p  # The vector of the segment starting at each ref point
        self.seg_len = np.linalg.norm(self.seg, axis=1)  # The length of the segments
        self.cumul_len = np.cumsum(self.seg_len) - self.seg_len  # The cumulative length from the start to the ref i
        self.full_len = self.cumul_len[-1] + self.seg_len[-1]  # The full length of the ref trajectory

    # Compute how much time (in ref_dts) the reference would take to go from two consecutive states projections
    def _compute_ref_time(self, projection):
        ref_id, delta = projection
        t = ref_id[1:] + delta[1:] - (ref_id[:-1] + delta[:-1])
        t[t < -self.ref_len / 2] += self.ref_len  # if the next step is behind the current point
        return t

    # compute the distance over the reference of two consecutive state projections
    def _compute_ref_distance(self, projection):
        ref_id, delta = projection
        partial_len_next = delta[1:] * self.seg_len[ref_id[1:]]
        partial_len_curr = delta[:-1] * self.seg_len[ref_id[:-1]]
        d = self.cumul_len[ref_id[1:]] + partial_len_next - (self.cumul_len[ref_id[:-1]] + partial_len_curr)
        d[d < -self.full_len / 2] += self.full_len
        return d

    '''
    Compute the minimum distance projection of each state over the reference as a couple (ref_id, delta)
    Projects the state on both the segments connected to the closest reference point and then select the closest projection
    '''
    def _compute_projection(self, state):
        _, ref_id = self.kdtree.query(state)
        ref_id = ref_id.squeeze()  # index of the next segment
        prev_ref_id = ref_id - 1  # index of the previous segment
        prev_ref_id[prev_ref_id == -1] = self.ref_len - 1  # if we got before the ref point 0
        ref_state = state - self.ref_p[ref_id]  # vector from the ref point to the state
        prev_ref_state = state - self.ref_p[prev_ref_id]
        delta = np.sum(self.seg[ref_id] * ref_state, axis=1) / np.square(self.seg_len[ref_id])  # <s-r,seg>/|seg|^2
        delta_prev = np.sum(self.seg[prev_ref_id] * prev_ref_state, axis=1) / np.square(self.seg_len[prev_ref_id])
        delta = delta.clip(0, 1);  # clips to points within the segment
        delta_prev = delta_prev.clip(0, 1);
        dist = np.linalg.norm(state - (self.ref_p[ref_id] + delta[:, None] * self.seg[ref_id]),
                              axis=1)  # point-segment distance
        dist_prev = np.linalg.norm(state - (self.ref_p[prev_ref_id] + delta_prev[:, None] * self.seg[prev_ref_id]),
                                   axis=1)
        closest = np.argmin(np.column_stack((dist, dist_prev)), axis=1)  # index of the one with minimum distance
        delta = np.column_stack((delta, delta_prev))[np.arange(delta.shape[0]), closest]  # select the one with minimum distance
        ref_id = np.column_stack((ref_id, prev_ref_id))[np.arange(ref_id.shape[0]), closest]
        return ref_id, delta


# returns how many centisecond of advantage the state projections gain relative to the reference
class Temporal_projection(Projection_reward):
    def __init__(self, ref_t, clip_range=None, ref_dt=1, sample_dt=10, penalty=None):
        super().__init__(ref_t, clip_range, ref_dt, sample_dt, penalty)

    def _compute_reward(self, data):
        state_p = data[['xCarWorld', 'yCarWorld']].values
        projection = self._compute_projection(state_p)
        ref_time = self._compute_ref_time(projection)
        reward = ref_time * self.ref_dt - self.sample_dt  # in centiseconds
        return reward

# returns the distance (meters) over the reference between two consecutive states projections
class Spatial_projection(Projection_reward):
    def __init__(self, ref_t, relative=True, clip_range=None, ref_dt=1, sample_dt=10, penalty=None):
        super().__init__(ref_t, clip_range, ref_dt, sample_dt, penalty)
        self.relative = relative

    def _compute_reward(self, data):
        state_p = data[['xCarWorld', 'yCarWorld']].values
        projection = self._compute_projection(state_p)
        d = self._compute_ref_distance(projection)
        if self.relative:  # d minus the distance the reference would have traveled in the same sample_dt time
            ref_id, delta = projection[0][:-1], projection[1][0:-1]
            ref_step = (self.sample_dt + delta) / self.ref_dt  # a step of sample_dt in reference terms
            ref_step_int = ref_step.astype(int)  # the reference gap after a step
            ref_id_next = ref_id + ref_step_int
            ref_id_next[ref_id_next >= self.ref_len] -= self.ref_len
            delta_next = ref_step - ref_step_int
            partial_len_next = delta_next * self.seg_len[ref_id_next]
            partial_len_curr = delta * self.seg_len[ref_id]
            ref_d = self.cumul_len[ref_id_next] + partial_len_next - (self.cumul_len[ref_id] + partial_len_curr)
            ref_d[ref_d < -self.full_len / 2] += self.full_len
            return d - ref_d
        return d


# returns the average speed (m/s) difference over two consecutive states projections between the sample and the reference
class Speed_projection(Projection_reward):
    def __init__(self, ref_t, relative=True, clip_range=None, ref_dt=1, sample_dt=10, penalty=None):
        super().__init__(ref_t, clip_range, ref_dt, sample_dt, penalty)
        self.relative = relative

    def _compute_reward(self, data):
        state_p = data[['xCarWorld', 'yCarWorld']].values
        projection = self._compute_projection(state_p)
        d = self._compute_ref_distance(projection)
        speed = (100 / self.sample_dt) * d  # in m/s
        if self.relative:
            t = self._compute_ref_time(projection) * self.ref_dt / 100
            ref_id = projection[0][:-1]
            ref_speed = np.empty_like(t)
            zeros = t == 0
            notz = np.logical_not(zeros)
            ref_speed[notz] = d[notz] / t[notz] #if t!=0
            ref_speed[zeros] = (100 / self.ref_dt) * self.seg_len[ref_id[zeros]]  # if t=0, the average speed on that segment
            return speed - ref_speed
        return speed


class Curv_temporal(Projection_reward):
    def __init__(self, ref_t, clip_range=None, ref_dt=1, sample_dt=10, penalty=None):  # dts in centiseconds
        self.ref_len = ref_t.shape[0]
        self.clipping = bool(clip_range)
        self.clip_range = clip_range
        self.ref_dt = ref_dt
        self.sample_dt = sample_dt

    def _compute_reward(self, data):
        state_p = data['ref_s'].values
        projection = state_p.astype(int), state_p - state_p.astype(int)
        ref_time = self._compute_ref_time(projection)
        reward = ref_time * self.ref_dt - self.sample_dt  # in centiseconds
        return reward


##### PENALIZATION TERM FOR REWARD #####
class RewardPenalty(object):
    """Class RewardPenalty is used to add a penalty term to the reward function."""
    def __init__(self):
        super(RewardPenalty, self).__init__()

    def compute_penalty(data):
        raise NotImplementedError

    def __call__(self, data):
        return self.compute_penalty(data)


class LikelihoodPenalty(RewardPenalty):
    """Class LikelihoodPenalty adds a penalty based to the likelihood of the state"""

    valid_kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']
    kernel_values = ['gaussian', 'epanechnikov', 'exponential']
    bandwidth_values = np.logspace(-1, 1, 20)

    def __init__(self, alpha=None, scale_f=0, kernel=None, bandwidth=None):
        super(LikelihoodPenalty, self).__init__()
        self.alpha = alpha
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.scale_f = scale_f

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        self._alpha = a

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, k):
        if k is None:
            self._kernel = k
        elif k not in LikelihoodPenalty.valid_kernels:
            raise Exception("Not valid kernel.")

        self._kernel = k

    @property
    def bandwidth(self):
        return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, b):
        self._bandwidth = b


    def tuning(self, X, params, random_state=1, n_jobs=1):
        np.random.seed(random_state)

        ids = np.arange(X.shape[0])
        np.random.shuffle(ids)
        X_shfl = X[ids, :]

        fixed_params = {}
        if self.bandwidth is not None:
            fixed_params['bandwidth'] = self.bandwidth
        if self.kernel is not None:
            fixed_params['kernel'] = self.kernel

        if fixed_params:
            search = GridSearchCV(KernelDensity(**fixed_params), param_grid=params, cv=10, n_jobs=n_jobs)
        else:
            search = GridSearchCV(KernelDensity(), param_grid=params, cv=10, n_jobs=n_jobs)

        search.fit(X_shfl)

        if 'bandwidth' in search.best_params_.keys():
            self.bandwidth = search.best_params_['bandwidth']

        if 'kernel' in search.best_params_.keys():
            self.kernel = search.best_params_['kernel']


    def fit(self, X, n_jobs=1):

        params = {}
        if self.bandwidth is None:
            params['bandwidth'] = LikelihoodPenalty.bandwidth_values
        if self.kernel is None:
            params['kernel'] = LikelihoodPenalty.kernel_values

        if (self.bandwidth is None) or (self.kernel is None):
            self.tuning(X, params, n_jobs)

        self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth).fit(X)

        if self.alpha is None:
            train_log = self.kde.score_samples(X)
            self.alpha = abs(1 / np.mean(train_log))
            self.scale_f = 1

    def compute_penalty(self, X):
        logp = self.kde.score_samples(X)
        return self.alpha * logp + self.scale_f

    def compute_offroad_penalty(self, X, trackPos):
        logp = self.kde.score_samples(X)
        mask = np.absolute(trackPos) > 1
        trackPos[~mask] = 0
        trackPos[mask] = -np.absolute(trackPos[mask])*1000#-50
        trackPos = np.clip(trackPos, a_min=-50, a_max=None)
        return self.alpha * logp + self.scale_f + trackPos


class LikelihoodPenaltyOffroad(LikelihoodPenalty):

    def __call__(self, data, trackPos):
        return self.compute_penalty(data, trackPos)

    def compute_penalty(self, X, trackPos):
        logp = self.kde.score_samples(X)
        mask = np.absolute(trackPos) > 1
        trackPos[~mask] = 0
        trackPos[mask] = -np.absolute(trackPos[mask])*20#-50
        trackPos = np.clip(trackPos, a_min=-50, a_max = None)
        return self.alpha * logp + self.scale_f + trackPos
