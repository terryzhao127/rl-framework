from io import BytesIO

import numpy as np

from .data_pb2 import Data


def arr2bytes(arr):
    arr_bytes = BytesIO()
    np.save(arr_bytes, arr, allow_pickle=False)
    return arr_bytes.getvalue()


def bytes2arr(arr_bytes):
    arr = np.load(BytesIO(arr_bytes), allow_pickle=False)
    return arr


class DataCollection:
    """
    Send data when an episode is done or the number of collected data equals `size`
    """

    def __init__(self, size):
        self.buffer = []
        self.next_state = None
        self.done = None
        self.size = size

    def push(self, state, action, value, neglogp, reward, next_state, done):
        self.buffer.append([state, action, value, neglogp, reward])
        self.next_state = next_state
        self.done = done

        if self.done or len(self.buffer) == self.size:
            items = list(zip(*self.buffer))
            states = arr2bytes(np.stack(items[0]))
            actions, values, neglogps, rewards = [arr2bytes(np.array(x).reshape(-1)) for x in items[1:]]
            next_state = arr2bytes(self.next_state)

            data = Data(states=states, actions=actions, values=values, neglogps=neglogps,
                        rewards=rewards, next_state=next_state, done=self.done)

            self.buffer = []
            self.next_state = None
            self.done = None

            return data.SerializeToString()
        else:
            return None


def parse_data(data_string):
    data = Data()
    data.ParseFromString(data_string)
    items = [data.states, data.actions, data.values, data.neglogps, data.rewards, data.next_state]
    items = [bytes2arr(item) for item in items]
    items.append(data.done)
    return items
