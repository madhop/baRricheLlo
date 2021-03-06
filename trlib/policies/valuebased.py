import numpy as np
from numpy import matlib
from trlib.policies.policy import Policy
from trlib.policies.qfunction import QFunction
from trlib.policies.double_qfunction import DoubleQFunction

class ValueBased(Policy):
    """
    A value-based policy is a policy that chooses actions based on their value.
    The action-space is always discrete for this kind of policy.
    """

    def __init__(self,actions,Q):

        self._actions = np.array(actions)
        self._n_actions = len(actions)
        self.Q = Q

    @property
    def actions(self):
        return self._actions

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self,value):

        if (not isinstance(value, QFunction)) & (not isinstance(value, DoubleQFunction)):
            raise TypeError("The argument must be a QFunction or DoubleQFunction")

        self._Q = value

    def __call__(self, state):
        """
        Computes the policy value in the given state

        Parameters
        ----------
        state: S-dimensional vector

        Returns
        -------
        An A-dimensional vector containing the probabilities pi(.|s)
        """
        raise NotImplementedError

    def _q_values(self, state):
        return self._Q.values(np.concatenate((matlib.repmat(state, self._n_actions, 1), self._actions), 1))
        #return self._Q.values(np.concatenate((matlib.repmat(state, self._n_actions, 1), self._actions[:,np.newaxis]), 1))

class EpsilonGreedy(ValueBased):
    """
    The epsilon-greedy policy.
    The parameter epsilon defines the probability of taking a random action.
    Set epsilon to zero to have a greedy policy.
    """

    def __init__(self,actions,Q,epsilon):

        super().__init__(actions, Q)
        self.epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self,value):
        if value < 0 or value > 1:
            raise AttributeError("Epsilon must be in [0,1]")
        self._epsilon = value

    def __call__(self, state):

        probs = np.ones(self._n_actions) * self._epsilon / self._n_actions
        probs[np.argmax(self._q_values(state))] += 1 - self._epsilon
        return probs

    def sample_action(self, state):

        if np.random.uniform() < self._epsilon:
            return np.array([self._actions[np.random.choice(self._n_actions)]])
        else:
            return np.array([self._actions[np.argmax(self._q_values(state))]])

class EpsilonGreedyNoise(EpsilonGreedy):
    """docstring for EpsilonGreedyNoise."""

    def __init__(self, actions, Q, epsilon, stdNoise):
        super(EpsilonGreedyNoise, self).__init__(actions, Q, epsilon)
        self.stdNoise = stdNoise

    @property
    def stdNoise(self):
        return self._stdNoise

    @stdNoise.setter
    def stdNoise(self,value):
        self._stdNoise = value

    def sample_action(self, state):
        a = np.array([self._actions[np.argmax(self._q_values(state))]])
        a[0][0] = a[0][0] + np.random.normal(0, self._stdNoise)
        a = np.clip(a, a_min=-1, a_max = 1)
        return a

class Softmax(ValueBased):
    """
    The softmax (or Boltzmann) policy.
    The parameter tau controls exploration (for tau close to zero the policy is almost greedy)
    """

    def __init__(self,actions,Q,tau):

        super().__init__(actions, Q)
        self.tau = tau

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self,value):
        if value <= 0:
            raise AttributeError("Tau must be strictly greater than zero")
        self._tau = value

    def __call__(self, state):

        num = self._q_values(state)
        exps = np.exp(num / self._tau)
        l = exps / np.sum(exps)
        idx = np.isnan(l)
        try:
            l[idx] = np.min(l[~idx])
        except Exception as e:
            l = [(1/len(l))]* len(l)
            print('l:', l)
        return list(l)

    def sample_action(self, state):

        return np.array([self._actions[np.random.choice(self._n_actions, p = self(state))]])
