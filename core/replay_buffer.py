import random
import numpy as np
from collections import deque


class ReplayBuffer:

    def __init__(self, size=0):
        if size > 0:
            self._storage = deque(maxlen=size)
        else:
            self._storage = deque()

    def __len__(self):
        return len(self._storage)

    def add(self, item):
        self._storage.append(item)

    def extend(self, items):
        self._storage.extend(items)

    def sample(self, batch_size):
        items = random.sample(self._storage, batch_size)
        items = [np.array(item) for item in zip(*items)]
        return items

    def all(self, clear=False):
        items = [item for item in self._storage]
        items = [np.array(item) for item in zip(*items)]
        if clear:
            self.clear()
        return items

    def clear(self):
        self._storage.clear()
