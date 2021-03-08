import random
from collections import defaultdict, deque
from typing import Dict, List

import numpy as np


class MemPool:

    def __init__(self, capacity: int = None, keys: List[str] = None) -> None:
        self._keys = keys
        self._sizes = deque(maxlen=capacity)
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

        self._sizes.append(data[self._keys[0]].shape[0])

    def sample(self, size: int = -1) -> Dict[str, np.ndarray]:
        """
        Sample training data from memory pool
        :param size: The number of sample data, default '-1' that indicates all data
        :return: The sampled and concatenated training data
        """

        num = len(self)
        indices = list(range(num))
        if 0 < size < num:
            indices = random.sample(indices, size)
        indices = np.array(indices)

        return {key: np.concatenate(self.data[key])[indices] for key in self._keys}

    def clear(self) -> None:
        """Clear all data"""
        for key in self._keys:
            self.data[key].clear()
        self._sizes.clear()

    def __len__(self):
        return sum(self._sizes)