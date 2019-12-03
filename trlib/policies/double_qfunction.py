import numpy as np
from numpy import matlib


class DoubleQFunction:
    """
    Base class for all Double Q-functions
    """

    def __init__(self, regressor_type, state_dim, action_dim, **regressor_params):

        self._regressor = [regressor_type(**regressor_params), regressor_type(**regressor_params)]
        self._state_dim = state_dim
        self._action_dim = action_dim

    def __call__(self, state, action):

        return self.values(np.concatenate((state,action),0)[np.newaxis,:])

    def values(self, sa, q_id=None):
        """
        Computes the values of all state-action vectors in sa. In double fqi the value is
        the mean of the values of the two Q functions.

        Parameters
        ----------
        sa: an Nx(S+A) matrix
        q_id: id of the regressor to use, if specified then it returns the Q of the ith regressor
              otherwise it computes the Q of both and then return the mean

        Returns
        -------
        An N-dimensional vector with the value of each state-action row vector in sa
        """
        if not np.shape(sa)[1] == self._state_dim + self._action_dim:
            raise AttributeError("An Nx(S+A) matrix must be provided")

        if q_id is None:

            Q = (self._regressor[0].predict(sa) + self._regressor[1].predict(sa)) / 2

            return Q
        else:
            return self._regressor[q_id].predict(sa)

    def max(self, states, actions=None, absorbing=None, q_id=None):
        """
        Computes the action among actions achieving the maximum value for each state in states.
        Where the value is the mean of the values of the two Q functions.

        Parameters:
        -----------
        states: an NxS matrix
        actions: a list of A-dimensional vectors
        absorbing: an N-dimensional vector specifying whether each state is absorbing
        q_id: id of the regressor

        Returns:
        --------
        An NxA matrix with the maximizing actions and an N-dimensional vector with their values
        """
        raise NotImplementedError

    def fit(self, sa, q, q_id=None, **fit_params):
        """
        Fit the regressors.

        Parameters:
        -----------
        sa: an Nx(S+A) matrix
        q: an Nx1 vector
        q_id: id of the regressor to fit
        fit_params: parameters of the fit

        """
        if q_id is None:
            for i in range(len(self._regressor)):
                self._regressor[i].fit(sa, q, **fit_params)
        else:
            self._regressor[q_id].fit(sa, q, **fit_params)


class DoubleFittedQ(DoubleQFunction):
    """
    A DoubleFittedQ is a Q-function represented by an underlying regressor
    that has been fitted on some data. The regressor receives SA-dimensional
    vectors and predicts their scalar value.
    The actions used during the maxQ evaluation are not the same as in DoubleFittedQ,
    but are specific for each state.
    """

    def max(self, states, actions=None, absorbing=None, q_id=None):
        """
        actions parameter is an instance of the object ActionDispatcher
        that with the method get_actions(s) returns the list of actions
        to use for the state s.
        """

        if not np.shape(states)[1] == self._state_dim:
            raise AttributeError("Wrong dimensions of the input matrices")
        if actions is None:
            raise AttributeError("Actions must be provided")

        # Get the number of action for each state
        n_actions_per_state = list(map(lambda x: len(x), map(lambda s: actions[tuple(s)], states)))
        tot_n_actions = sum(n_actions_per_state)
        n_states = np.shape(states)[0]

        sa = np.empty((tot_n_actions, self._state_dim + self._action_dim))

        absorbing_ids = []
        end = 0
        for i in range(n_states):
            # set state interval variables
            start = end
            end = end + n_actions_per_state[i]

            # populate the matrix with the ith state prime
            sa[start:end, 0:self._state_dim] = matlib.repmat(states[i, :], n_actions_per_state[i], 1)

            # populate the matrix with the actions of the action set of ith state prime
            sa[start:end, self._state_dim:] = \
                np.array(actions[tuple(states[i, :])]).reshape((n_actions_per_state[i], self._action_dim))

            if absorbing is not None:
                if absorbing[i]:
                    absorbing_ids.extend(range(start, end))

        # Compute values of Q
        vals = self.values(sa, q_id)

        # set to 0 the Q of absorbing states
        vals[absorbing_ids] = 0

        max_vals = np.empty(n_states)
        max_actions = np.empty((n_states,self._action_dim))

        end = 0
        for i in range(n_states):
            # set state interval variables
            start = end
            end = end + n_actions_per_state[i]

            val = vals[start:end]
            a = np.argmax(val)
            max_vals[i] = val[a]

            max_actions[i, :] = actions[tuple(states[i, :])][a]

        return max_vals, max_actions

    def max_sa(self, sa, n_actions_per_state, absorbing=None, q_id=None):
        """ Return the best action for each state.

        Input:
            - sa: state action matrix, it contains for each state the pair of s with every action
            in its action set. Shape: [sum(n_actions_per_state), state_dim + action_dim]
        """

        # compute Q values for each state action pair
        vals = self.values(sa, q_id)

        n_states = len(n_actions_per_state)

        max_vals = np.empty(n_states)
        max_actions = np.empty((n_states, self._action_dim))

        end = 0
        for i in range(n_states):
            # set state interval variables
            start = end
            end = end + n_actions_per_state[i]

            # get state actions
            state_actions = sa[start:end, self._state_dim:]

            if absorbing is not None:
                # if the state is absorbing return Q = 0
                if absorbing[i]:
                    max_vals[i] = 0
                    max_actions[i, :] = state_actions[0, :]
                else:
                    val = vals[start:end]
                    a = np.argmax(val)
                    max_vals[i] = val[a]

                    max_actions[i, :] = state_actions[a, :]
            else:
                val = vals[start:end]
                a = np.argmax(val)
                max_vals[i] = val[a]

                max_actions[i, :] = state_actions[a, :]

        return max_vals, max_actions
