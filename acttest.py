#acttest.py
import os
from data_pb2 import Data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

@pytest.mark.skip
def define_model(state_size, action_size):
    model = Sequential()
    model.add(Conv2D(16, 8, 4, activation='relu', input_shape=state_size))
    model.add(Conv2D(32, 4, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    return model

# actor端100个step后会在本地存字符串'100 message send'
# 同时将其发送到learner，learner也会将其存在本地
@pytest.mark.finished
def test_data_send(path = 'data.out'):
    data_file = None
    data = Data()
    if os.path.exists(path):
        data_file = open(path)
    else:
        print('No data send output!')

    if data_file != None:
        data_file = data_file.encode('ascii')
        data.ParseFromString(data_file)
    else:
        print('Data output None!')

    # 手册
    # if isinstance(data.state, str) and isinstance(data.action, int) and isinstance(data.reward, float) and isinstance(data.next_state, str) and isinstance(data.done, bool):
    #     print('Data type right')
    # else:
    #     print('Data type wrong!')

@pytest.mark.finished
def test_model_update(path = 'save/'):
    if os.path.exists(path):
        print('Model saved!')
    else:
        print('No model save!')

    try:
        print('Model load!')
    except:
        print('Can\'t load model!')