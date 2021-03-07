import random
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np


class MemPool:
    _not_initialized_error = RuntimeError('MemPool is not initialized yet')

    def __init__(self, capacity: int = None, keys: List[str] = None) -> None:
        self._keys = keys

        if keys is None:
            self.data = defaultdict(lambda: deque(maxlen=capacity))
        else:
            self.data = {key: deque(maxlen=capacity) for key in keys}

    def push(self, data: Dict[str, np.ndarray]) -> None:
        """Push data into memory pool"""
        for key, value in data.items():
            self.data[key].append(value)

        if self._keys is None:
            self._keys = list(self.data.keys())

    def get_sample_size(self):
        try:
            return sum([d.shape[0] for d in self.data[self.keys()[0]]])
        except RuntimeError:
            # MemPool is not initialized yet
            return 0

    def sample(self, size: int = -1) -> Dict[str, np.ndarray]:
        """
        Sample training data from memory pool
        :param size: The number of sample data, default '-1' that indicates all data
        :return: The sampled and concatenated training data
        """

        indices = list(range(self.get_sample_size()))
        if size != -1:
            indices = random.sample(indices, size)
        indices = np.array(indices)

        if len(self) == 1:
            return {key: self.data[key][0][indices] for key in self.keys()}
        else:
            return {key: np.concatenate(self.data[key])[indices] for key in self.keys()}

    def clear(self) -> None:
        """Clear all data"""
        for key in self.keys():
            self.data[key].clear()

    def keys(self) -> List[str]:
        """Get data keys in memory pool"""
        if self._keys is None:
            raise MemPool._not_initialized_error

        return self._keys

    def __len__(self):
        return len(self.data[self.keys()[0]])
