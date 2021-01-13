import random
import numpy as np
from collections import deque


class ReplayBuffer:

    def __init__(self, size):
        self._storage = deque(maxlen=size)

    def __len__(self):
        return len(self._storage)

    def add(self, item):
        self._storage.append(item)

    def sample(self, batch_size):
        items = random.sample(self._storage, batch_size)
        items = [np.array(item) for item in zip(*items)]
        return items
