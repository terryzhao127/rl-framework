#acttest.py
import os
from data_pb2 import Data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

def define_model(state_size, action_size):
    model = Sequential()
    model.add(Conv2D(16, 8, 4, activation='relu', input_shape=state_size))
    model.add(Conv2D(32, 4, 2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    return model

def test_data_send(path):
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

def test_model_save(state_size, action_size, path):
    if os.path.exists(path):
        print('Model saved!')
    else:
        print('No model save!')

    actor = define_model(state_size, action_size)

    try:
        actor.load_weights(path)
        print('Model load!')
    except:
        print('Can\'t load model!')

if __name__ == '__main__':
    test_data_send('framework/dataout.out')
    test_model_save()