from pyarrow import serialize, deserialize


class DataCollection:
    """
    Send data when an episode is done or the number of collected data equals `size`
    """

    def __init__(self, size):
        self.size = size
        self.cnt = 0
        self.data = None
        self.next_state = None
        self.done = None

    def push(self, state, reward, next_state, done, act_data):
        act_data.update({'state': state, 'reward': reward})
        if self.data is None:
            self.data = {key: [val] for key, val in act_data.items()}
        else:
            for key, val in act_data.items():
                self.data[key].append(val)
        self.cnt += 1
        self.next_state = next_state
        self.done = done

        if self.done or self.cnt == self.size:
            self.data.update({'next_state': self.next_state, 'done': self.done})
            data = serialize(self.data).to_buffer()

            self.cnt = 0
            self.data = None
            self.next_state = None
            self.done = None

            return data
        else:
            return None


def parse_data(data_string):
    return deserialize(data_string)
