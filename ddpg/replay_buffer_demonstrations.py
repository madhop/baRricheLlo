import pandas as pd
import numpy as np

from stable_baselines.deepq.replay_buffer import PrioritizedReplayBuffer
from stable_baselines.common.math_util import scale_action

class PrioritizedReplayBufferDemonstrations(PrioritizedReplayBuffer):
    """docstring for PrioritizedReplayBufferDemonstrations."""

    def __init__(self, buffer_size, alpha, demonstrations, action_space):
        """
        Create Prioritized Replay buffer.

        See Also PrioritizedReplayBuffer.__init__

        :param buffer_size: (int) Max number of non-demonstrations transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        :param demonstrations: (list) transizions from demonstrations. These will never leave the buffer
        """
        self.d_size = len(demonstrations)
        super(PrioritizedReplayBufferDemonstrations, self).__init__(self.d_size + buffer_size, alpha)
        for t in demonstrations:
            scaled_a = scale_action(action_space, t[1])
            super().add(t[0], scaled_a, *t[2:])


    def update_priorities(self, batch_idxes, td_errors):
        new_priorities = np.power(td_errors, 2) + self.prioritized_replay_eps
        new_priorities[batch_idxes < self.d_size] = new_priorities[batch_idxes < self.d_size] + self.prioritized_replay_eps_D
        super.update_priorities(batch_idxes, new_priorities)

    def add(self, obs_t, action, reward, obs_tp1, done):
        super().add(obs_t, action, reward, obs_tp1, done)
        # When the buffer is full, delete the oldest non-demonstrations transition
        if self._next_idx <= self.d_size:
            self._next_idx = self.d_size
