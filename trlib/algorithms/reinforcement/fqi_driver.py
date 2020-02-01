import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KDTree
from trlib.algorithms.algorithm import Algorithm
from trlib.policies.qfunction import FittedQ
from trlib.policies.double_qfunction import DoubleFittedQ
from trlib.utilities.interaction import split_data
import time
import math
from trlib.utilities.ActionDispatcher import ConstantActionDispatcher
from numpy import matlib


class FQIDriver(Algorithm):
    """
    Fitted Q-Iteration

    References
    ----------
      - Ernst, Damien, Pierre Geurts, and Louis Wehenkel
        Tree-based batch mode reinforcement learning
        Journal of Machine Learning Research 6.Apr (2005): 503-556
    """

    def __init__(self, mdp, policy, actions, max_iterations, regressor_type, data, action_dispatcher, state_mask,
                 data_mask, s_norm, filter_a_outliers, ad_n_jobs, ad_param, verbose=False, precompute_sprime_a=True,
                 **regressor_params):
        """

        :param mdp: instance of TrackEnv
        :param policy: instance of ValueBased class
        :param actions: list of all actions i.e., dataset[action_cols].values
        :param max_iterations: number of iteration of the algorithm
        :param regressor_type: regressor to use
        :param data: list of tsarsa tuples i.e., dataset[cols].values
        :param action_dispatcher: action dispatcher class
        :param s_norm: bool, True to normalize the states in the action dispatcher
        :param filter_a_outliers: bool, True to filter the outliers in the action set for each state
        :param ad_n_jobs: number of parallel jobs for the action dispatcher
        :param ad_param: parameter of the action dispatcher
        :param verbose: True to verbose computation
        :param precompute_sprime_a: True to precompute the matrix state_prime-action used in maxQ
        :param regressor_params: parameters of the regressor
        """

        super().__init__("FQI Driver", mdp, policy, verbose)

        self._data = data
        self.state_dim = mdp.state_dim
        self.sprime_a = []
        self.n_actions_per_state_prime = []
        self.precompute_sprime_a = precompute_sprime_a

        self._actions = {}
        self.action_dim = mdp.action_dim

        self._action_dispatcher = self.create_action_dispatcher(action_dispatcher, actions, state_mask, data_mask,
                                                                s_norm, filter_a_outliers, ad_n_jobs, ad_param)

        self._max_iterations = max_iterations
        self._regressor_type = regressor_type

        self._policy.Q = FittedQ(regressor_type, mdp.state_dim, mdp.action_dim, **regressor_params)

        self.reset()

    def create_action_dispatcher(self, action_dispatcher, actions, state_mask, data_mask, s_norm, filter_a_outliers,
                                 ad_n_jobs, ad_param):
        """
        :param action_dispatcher: class of action dispatcher
        :param actions: list of all the actions
        :param state_mask: ids of the state to consider
        :param data_mask: ids of the data matrix to consider. Data matrix contains t|s|a|r|s'|abs
        :param s_norm: True to normalize the states
        :param filter_a_outliers: True to filter the outliers
        :param ad_n_jobs: number of parallel jobs
        :param ad_param: parameter of the action dispatcher
        :return:
        """

        if action_dispatcher == ConstantActionDispatcher:
            ad = action_dispatcher(actions)
        else:
            knn_data = self._data[:, data_mask]
            actions = list(map(lambda x: list(x), actions))

            if s_norm:
                s_scaler = MinMaxScaler()
                knn_data = s_scaler.fit_transform(knn_data)
            else:
                s_scaler = None

            if filter_a_outliers:
                a_scaler = MinMaxScaler().fit(actions)
            else:
                a_scaler = None

            state_kdt = KDTree(knn_data)

            ad = action_dispatcher(actions, state_mask, state_kdt, s_scaler, a_scaler, filter_a_outliers, ad_n_jobs,
                                   ad_param)
        return ad

    def _iter(self, sa, r, s_prime, absorbing, **fit_params):

        self.display("Iteration {0}".format(self._iteration))

        if self._iteration == 0:
            y = r
            maxq_time = 0
        else:
            tm = time.time()

            if len(self.sprime_a) > 0:
                maxq, _ = self._policy.Q.max_sa(self.sprime_a, self.n_actions_per_state_prime, absorbing)
            else:
                maxq, _ = self._policy.Q.max(s_prime, self._actions, absorbing)

            y = r.ravel() + self._mdp.gamma * maxq
            maxq_time = time.time() - tm
            print('maxQ {}'.format(maxq_time))

        tf = time.time()
        self._policy.Q.fit(sa, y.ravel(), **fit_params)
        fit_time = time.time() - tf
        print('fitQ {}'.format(fit_time))
        self._iteration += 1
        return maxq_time, fit_time

    def _step_core(self, **kwargs):

        self._iteration = 0

        _,_,_,r,s_prime,absorbing,sa = split_data(self._data, self._mdp.state_dim, self._mdp.action_dim)

        # compute the action set for each state prime
        self.compute_action_set(s_prime)

        if self.precompute_sprime_a:
            ts = time.time()
            # compute the matrix state-action for each state prime and for each action in its action set
            self.compute_s_prime_a(s_prime)
            print('Time for sprime a mat {}'.format(time.time() - ts))

        time_list = []
        maxq_time_list = []
        fit_time_list = []
        for i in range(self._max_iterations):

            maxq_time, fit_time = self._iter(sa, r, s_prime, absorbing, **kwargs)

            maxq_time_list = maxq_time_list + [maxq_time]
            fit_time_list = fit_time_list + [fit_time]
            time_list = time_list + [maxq_time + fit_time]
            print('Elapsed time {}'.format(maxq_time + fit_time))

        print('Total elapsed time: {}'.format(np.sum(time_list)))
        print('Mean elapsed time: {}'.format(np.mean(time_list)))
        print('Std elapsed time: {}'.format(np.std(time_list)))

        # reset sprime_a to save memory
        self.sprime_a = []
        self.n_actions_per_state_prime = []

        self._result.update_step(n_episodes = self.n_episodes, n_samples = self._data.shape[0])
        self._result.add_fields(elapsed_time=time_list, maxq_time = maxq_time_list, fit_time=fit_time_list)

    def compute_action_set(self, s):
        # Compute for the given dataset the list of actions to use for the maxQ for each state prime
        print('Finding nearest actions for each state prime')
        self._actions = {}

        t1 = time.time()
        action_list = self._action_dispatcher.get_actions(s)
        print('Time for action list {}'.format(time.time() - t1))

        ts = time.time()
        for i in range(len(s)):
            self._actions[tuple(s[i, :])] = action_list[i]
        print('Time for action set {}'.format(time.time() - ts))

    def compute_s_prime_a(self, s_primes):
        """Given the list of next states s_prime, it returns the matrix state actions that
        for each state prime the contains all the pairs s'a where a is in the action set
        of s'.
        """

        # Get the number of actions for each state
        n_actions_per_state = list(map(lambda x: len(x), map(lambda s: self._actions[tuple(s)], s_primes)))
        tot_n_actions = sum(n_actions_per_state)
        n_states = s_primes.shape[0]
        sa = np.empty((tot_n_actions, self.state_dim + self.action_dim))

        end = 0
        for i in range(n_states):
            # set interval variables
            start = end
            end = end + n_actions_per_state[i]

            # set state prime
            i_s_prime = s_primes[i, :]
            n_actions = n_actions_per_state[i]

            # populate the matrix with the ith state prime
            sa[start:end, 0:self.state_dim] = matlib.repmat(i_s_prime, n_actions, 1)

            # populate the matrix with the actions of the action set of ith state prime
            sa[start:end, self.state_dim:] =\
                np.array(self._actions[tuple(i_s_prime)]).reshape((n_actions, self.action_dim))

        # reset self._actions to save memory
        self._actions = []

        self.sprime_a = sa
        self.n_actions_per_state_prime = n_actions_per_state

    def reset(self):

        super().reset()

        self._iteration = 0
        self.sprime_a = []
        self.n_actions_per_state_prime = []

        self._result.add_fields(max_iterations=self._max_iterations,
                                regressor_type=str(self._regressor_type.__name__),
                                policy = str(self._policy.__class__.__name__))


class DoubleFQIDriver(Algorithm):
    """DoubleFQIDriver is the variation of the FQI which learns two different
    Q functions during the iterations.
    """

    def __init__(self, mdp, policy, actions, max_iterations, regressor_type, data, action_dispatcher, state_mask,
                 data_mask, s_norm, filter_a_outliers, ad_n_jobs, ad_param, verbose=False, precompute_sprime_a=True,
                 **regressor_params):
        """

        :param mdp: instance of TrackEnv
        :param policy: instance of ValueBased class
        :param actions: list of all actions i.e., dataset[action_cols].values
        :param max_iterations: number of iteration of the algorithm
        :param regressor_type: regressor to use
        :param data: list of tsarsa tuples i.e., dataset[cols].values
        :param action_dispatcher: action dispatcher class
        :param s_norm: bool, True to normalize the states in the action dispatcher
        :param filter_a_outliers: bool, True to filter the outliers in the action set for each state
        :param ad_n_jobs: number of parallel jobs for the action dispatcher
        :param ad_param: parameter of the action dispatcher
        :param verbose: True to verbose computation
        :param precompute_sprime_a: True to precompute the matrix state_prime-action used in maxQ
        :param regressor_params: parameters of the regressor
        """
        super().__init__("FQI Driver", mdp, policy, verbose)

        # Split the dataset it two disjoint subsets
        n_samples = data.shape[0]
        ids = list(range(n_samples))
        np.random.seed(42)
        np.random.shuffle(ids)
        self._ids_A = ids[:math.floor(n_samples / 2)]
        self._ids_B = ids[math.floor(n_samples / 2):]

        self._data = data
        self._data_A = data[self._ids_A, :]
        self._data_B = data[self._ids_B, :]

        self.state_dim = mdp.state_dim
        self.sprime_a_A = []
        self.sprime_a_B = []
        self.n_actions_per_state_prime_A = []
        self.n_actions_per_state_prime_B = []
        self.precompute_sprime_a = precompute_sprime_a

        # Initialize the action dictionaries and create the action dispatcher objects
        self._actions_A = {}
        self._actions_B = {}
        self.action_dim = mdp.action_dim
        self._action_dispatcher, self._action_dispatcher_A, self._action_dispatcher_B =\
            self.create_action_dispatcher(action_dispatcher, actions, state_mask, data_mask, s_norm, filter_a_outliers,
                                          ad_n_jobs, ad_param)

        self._max_iterations = max_iterations
        self._regressor_type = regressor_type

        self._policy.Q = DoubleFittedQ(regressor_type, mdp.state_dim, mdp.action_dim, **regressor_params)

        self.reset()

    def create_action_dispatcher(self, action_dispatcher, actions, state_mask, data_mask, s_norm, filter_a_outliers,
                                 ad_n_jobs, ad_param):
        """
        :param action_dispatcher: class of action dispatcher
        :param actions: list of all the actions
        :param state_mask: ids of the state to consider
        :param data_mask: ids of the data matrix to consider. Data matrix contains t|s|a|r|s'|abs
        :param s_norm: True to normalize the states
        :param filter_a_outliers: True to filter the outliers
        :param ad_n_jobs: number of parallel jobs
        :param ad_param: parameter of the action dispatcher
        :return:
        """

        actions_A = actions[self._ids_A, :]
        actions_B = actions[self._ids_B, :]

        if action_dispatcher == ConstantActionDispatcher:
            ad = action_dispatcher(actions)
            ad_A = action_dispatcher(actions_A)
            ad_B = action_dispatcher(actions_B)
        else:
            knn_data = self._data[:, data_mask]
            knn_data_A = self._data_A[:, data_mask]
            knn_data_B = self._data_B[:, data_mask]

            actions = list(map(lambda x: list(x), actions))
            actions_A = list(map(lambda x: list(x), actions_A))
            actions_B = list(map(lambda x: list(x), actions_B))

            if s_norm:
                s_scaler = MinMaxScaler()
                knn_data = s_scaler.fit_transform(knn_data)
                s_scaler_A = MinMaxScaler()
                knn_data_A = s_scaler_A.fit_transform(knn_data_A)
                s_scaler_B = MinMaxScaler()
                knn_data_B = s_scaler_B.fit_transform(knn_data_B)
            else:
                s_scaler = None
                s_scaler_A = None
                s_scaler_B = None

            if filter_a_outliers:
                a_scaler = MinMaxScaler().fit(actions)
                a_scaler_A = MinMaxScaler().fit(actions_A)
                a_scaler_B = MinMaxScaler().fit(actions_B)
            else:
                a_scaler = None
                a_scaler_A = None
                a_scaler_B = None

            state_kdt = KDTree(knn_data)
            state_kdt_A = KDTree(knn_data_A)
            state_kdt_B = KDTree(knn_data_B)

            ad = action_dispatcher(actions, state_mask, state_kdt, s_scaler, a_scaler, filter_a_outliers,
                                   ad_n_jobs, ad_param)

            ad_A = action_dispatcher(actions_A, state_mask, state_kdt_A, s_scaler_A, a_scaler_A, filter_a_outliers,
                                     ad_n_jobs, ad_param)

            ad_B = action_dispatcher(actions_B, state_mask, state_kdt_B, s_scaler_B, a_scaler_B, filter_a_outliers,
                                     ad_n_jobs, ad_param)
        return ad, ad_A, ad_B

    def _iter(self, sa, r, s_prime, absorbing, **fit_params):
        """In double fqi we learn two Q functions. The procedure is the following:
        - split the dataset in two disjoint subsets
        - find the best actions for each s' of the two Q
        - fit the two models with half dataset
        """
        self.display("Iteration {0}".format(self._iteration))

        # 1) split the dataset
        sa_A = sa[self._ids_A, :]
        sa_B = sa[self._ids_B, :]
        r_A = r[self._ids_A]
        r_B = r[self._ids_B]
        s_prime_A = s_prime[self._ids_A, :]
        s_prime_B = s_prime[self._ids_B, :]
        absorbing_A = absorbing[self._ids_A].astype(bool)
        absorbing_B = absorbing[self._ids_B].astype(bool)

        # 2) compute the target
        if self._iteration == 0:
            y_A = r_A
            y_B = r_B
            maxq_time = 0

        else:

            tm = time.time()

            if len(self.sprime_a_A) > 0:

                # Find the best action
                _, maxa_A = self._policy.Q.max_sa(self.sprime_a_A, self.n_actions_per_state_prime_A, absorbing_A, 0)
                _, maxa_B = self._policy.Q.max_sa(self.sprime_a_B, self.n_actions_per_state_prime_B, absorbing_B, 1)

                # Get the value of the action using the other Q
                maxq_A = self._policy.Q.values(np.concatenate((s_prime_A, maxa_A), axis=1), 1)
                maxq_B = self._policy.Q.values(np.concatenate((s_prime_B, maxa_B), axis=1), 0)

                # Set to 0 the values of the absorbing states
                maxq_A[absorbing_A] = 0
                maxq_B[absorbing_B] = 0

            else:
                # Find the best action
                _, maxa_A = self._policy.Q.max(s_prime_A, self._actions_A, absorbing_A, 0)
                _, maxa_B = self._policy.Q.max(s_prime_B, self._actions_B, absorbing_B, 1)

                # Get the value of the action using the other Q
                maxq_A = self._policy.Q.values(np.concatenate((s_prime_A, maxa_A), axis=1), 1)
                maxq_B = self._policy.Q.values(np.concatenate((s_prime_B, maxa_B), axis=1), 0)

                # Set to 0 the values of the absorbing states
                maxq_A[absorbing_A] = 0
                maxq_B[absorbing_B] = 0

            # Compute the target
            y_A = r_A.ravel() + self._mdp.gamma * maxq_A
            y_B = r_B.ravel() + self._mdp.gamma * maxq_B
            maxq_time = time.time() - tm
            print('maxQ {}'.format(maxq_time))

        tf = time.time()

        self._policy.Q.fit(sa_A, y_A.ravel(), 0, **fit_params)
        self._policy.Q.fit(sa_B, y_B.ravel(), 1, **fit_params)

        fit_time = time.time() - tf
        print('fitQ {}'.format(fit_time))
        self._iteration += 1
        return maxq_time, fit_time

    def _step_core(self, **kwargs):

        self._iteration = 0

        _, _, _, r, s_prime, absorbing, sa = split_data(self._data, self._mdp.state_dim, self._mdp.action_dim)

        # compute the action set for each state prime
        self.compute_action_set(s_prime)

        if self.precompute_sprime_a:
            ts = time.time()
            # compute the matrix state-action for each state prime and for each action in its action set
            self.compute_s_prime_a(s_prime)
            print('Time for sprime a mat {}'.format(time.time() - ts))

        time_list = []
        maxq_time_list = []
        fit_time_list = []
        for i in range(self._max_iterations):
            maxq_time, fit_time = self._iter(sa, r, s_prime, absorbing, **kwargs)


            ## save policy pickle at each iteration
            #if (self._iteration+1)%5 == 0:
            if self._iteration+1 == 45 or self._iteration+1 == 80:
                policy_pickle_name = './model_file/Policies/Policy_' + str(self._iteration) + '.pkl'
                print('Save Q')
                with open(policy_pickle_name, 'wb') as output:
                    pickle.dump(self._policy, output, pickle.HIGHEST_PROTOCOL)


            maxq_time_list = maxq_time_list + [maxq_time]
            fit_time_list = fit_time_list + [fit_time]
            time_list = time_list + [maxq_time + fit_time]
            print('Elapsed time {}'.format(maxq_time + fit_time))

        print('Total elapsed time: {}'.format(np.sum(time_list)))
        print('Mean elapsed time: {}'.format(np.mean(time_list)))
        print('Std elapsed time: {}'.format(np.std(time_list)))

        self.sprime_a_A = []
        self.sprime_a_B = []
        self.n_actions_per_state_prime_A = []
        self.n_actions_per_state_prime_B = []

        self._result.update_step(n_episodes=self.n_episodes, n_samples=self._data.shape[0])
        self._result.add_fields(elapsed_time=time_list, maxq_time=maxq_time_list, fit_time=fit_time_list)

    def compute_action_set(self, s):
        # Compute for the given dataset the list of actions to use for the maxQ for each state prime
        print('Finding nearest actions for each state prime')

        s_A = s[self._ids_A, :]
        s_B = s[self._ids_B, :]

        t1 = time.time()
        action_list_A = self._action_dispatcher_A.get_actions(s_A)
        action_list_B = self._action_dispatcher_B.get_actions(s_B)
        print('Time for action list {}'.format(time.time() - t1))

        ts = time.time()
        for i in range(len(s_A)):
            self._actions_A[tuple(s_A[i, :])] = action_list_A[i]

        for i in range(len(s_B)):
            self._actions_B[tuple(s_B[i, :])] = action_list_B[i]

        print('Time for action set {}'.format(time.time() - ts))

    def compute_s_prime_a(self, s_primes):
        """Given the list of next states s_prime, it returns the matrix state actions that
        for each state prime the contains all the pairs s'a where a is in the action set
        of s'.
        """

        s_primes_A = s_primes[self._ids_A, :]
        s_primes_B = s_primes[self._ids_B, :]

        # Get the number of actions for each state
        n_actions_per_state_A = list(map(lambda x: len(x), map(lambda s: self._actions_A[tuple(s)], s_primes_A)))
        n_actions_per_state_B = list(map(lambda x: len(x), map(lambda s: self._actions_B[tuple(s)], s_primes_B)))

        tot_n_actions_A = sum(n_actions_per_state_A)
        tot_n_actions_B = sum(n_actions_per_state_B)

        n_states_A = s_primes_A.shape[0]
        n_states_B = s_primes_B.shape[0]

        sa_A = np.empty((tot_n_actions_A, self.state_dim + self.action_dim))
        sa_B = np.empty((tot_n_actions_B, self.state_dim + self.action_dim))

        end = 0
        for i in range(n_states_A):
            # set interval variables
            start = end
            end = end + n_actions_per_state_A[i]

            # set state prime
            i_s_prime = s_primes_A[i, :]
            n_actions = n_actions_per_state_A[i]

            # populate the matrix with the ith state prime
            sa_A[start:end, 0:self.state_dim] = matlib.repmat(i_s_prime, n_actions, 1)

            # populate the matrix with the actions of the action set of ith state prime
            sa_A[start:end, self.state_dim:] = \
                np.array(self._actions_A[tuple(i_s_prime)]).reshape((n_actions, self.action_dim))

        end = 0
        for i in range(n_states_B):
            # set interval variables
            start = end
            end = end + n_actions_per_state_B[i]

            # set state prime
            i_s_prime = s_primes_B[i, :]
            n_actions = n_actions_per_state_B[i]

            # populate the matrix with the ith state prime
            sa_B[start:end, 0:self.state_dim] = matlib.repmat(i_s_prime, n_actions, 1)

            # populate the matrix with the actions of the action set of ith state prime
            sa_B[start:end, self.state_dim:] = \
                np.array(self._actions_B[tuple(i_s_prime)]).reshape((n_actions, self.action_dim))

        # reset self._actions to save memory
        self._actions_A = []
        self._actions_B = []

        self.sprime_a_A = sa_A
        self.sprime_a_B = sa_B
        self.n_actions_per_state_prime_A = n_actions_per_state_A
        self.n_actions_per_state_prime_B = n_actions_per_state_B

    def reset(self):

        super().reset()
        self.sprime_a_A = []
        self.sprime_a_B = []
        self.n_actions_per_state_prime_A = []
        self.n_actions_per_state_prime_B = []
        self._iteration = 0

        self._result.add_fields(max_iterations=self._max_iterations,
                                regressor_type=str(self._regressor_type.__name__),
                                policy=str(self._policy.__class__.__name__))
