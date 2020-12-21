import numpy as np


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add_batch(self, items):
        for item in items:
            self.add(item)

    def add(self, item):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            self._storage[self._next_idx] = item
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices):
        items = [self._storage[i] for i in indices]
        items = [np.array(item) for item in zip(*items)]
        return items

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=batch_size)
        return self._encode_sample(indices)
