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
        print('PRBD Initialized')


    def update_priorities(self, batch_idxes, new_priorities, td_errors):
        new_priorities = np.abs(td_errors) + self.prioritized_replay_eps + self.prioritized_replay_eps_D
        # TODO: different update for demonstration and non-demonstration
        super.update_priorities(batch_idxes, new_priorities)

    def add(self, obs_t, action, reward, obs_tp1, done):
        super().add(obs_t, action, reward, obs_tp1, done)
        # When the buffer is full, delete the oldest non-demonstrations transition
        if self._next_idx <= self.d_size:
            self._next_idx = self.d_size
