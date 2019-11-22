import numpy as np
from numpy import matlib


class QFunction:
    """
    Base class for all Q-functions
    """

    def __call__(self, state, action):
        """
        Computes the value of the given state-action couple
        
        Parameters
        ----------
        state: an S-dimensional vector
        action: an A-dimensional vector
        
        Returns
        -------
        The value of (state,action).
        """
        raise NotImplementedError
        
    def values(self, sa):
        """
        Computes the values of all state-action vectors in sa
        
        Parameters
        ----------
        sa: an Nx(S+A) matrix
        
        Returns
        -------
        An N-dimensional vector with the value of each state-action row vector in sa
        """
        raise NotImplementedError
    
    def max(self, states, actions=None, absorbing=None):
        """
        Computes the action among actions achieving the maximum value for each state in states
        
        Parameters:
        -----------
        states: an NxS matrix
        actions: a list of A-dimensional vectors
        absorbing: an N-dimensional vector specifying whether each state is absorbing
        
        Returns:
        --------
        An NxA matrix with the maximizing actions and an N-dimensional vector with their values
        """
        raise NotImplementedError


class ZeroQ(QFunction):
    """
    A QFunction that is zero for every state-action couple.
    """
    
    def __call__(self, state, action):
        return 0
    
    def values(self, sa):
        return np.zeros(np.shape(sa)[0])


class FittedQ(QFunction):
    """
    A FittedQ is a Q-function represented by an underlying regressor
    that has been fitted on some data. The regressor receives SA-dimensional
    vectors and predicts their scalar value.
    The actions used during the maxQ evaluation are taken from a dictionary that
    for each states returns the action set
    """

    def __init__(self, regressor_type, state_dim, action_dim, **regressor_params):
        
        self._regressor = regressor_type(**regressor_params)
        self._state_dim = state_dim
        self._action_dim = action_dim
        
    def __call__(self, state, action):
        
        return self.values(np.concatenate((state,action),0)[np.newaxis,:])
    
    def values(self, sa):
        
        if not np.shape(sa)[1] == self._state_dim + self._action_dim:
            raise AttributeError("An Nx(S+A) matrix must be provided")
        return self._regressor.predict(sa)
    
    def max(self, states, actions=None, absorbing=None):
        """
        actions parameter is a dictionary with keys the states
        and values the action list
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
            sa[start:end, self._state_dim:] =\
                np.array(actions[tuple(states[i, :])]).reshape((n_actions_per_state[i], self._action_dim))
            if absorbing is not None:
                if absorbing[i]:
                    absorbing_ids.extend(range(start, end))
        
        # compute values of Q  
        vals = self.values(sa)
        
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
   
    def max_sa(self, sa, n_actions_per_state, absorbing=None):
        """ Return the best action for each state.
        
        Input:
            - sa: state action matrix, it contains for each state the pair of s with every action
            in its action set. Shape: [sum(n_actions_per_state), state_dim + action_dim]
        """
        
        # compute Q values for each state action pair
        vals = self.values(sa)
        
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
        
    def fit(self, sa, q, **fit_params):
        
        self._regressor.fit(sa, q, **fit_params)

