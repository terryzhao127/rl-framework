#leatest.py
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

@pytest.mark.finished
def test_data_receive():
    data_file = None
    data = Data()

    assert os.path.exists('test/data.out'):
    data_file = open('test/data.out')

    assert data_file != None:
    data_file = data_file.encode('ascii')
    data.ParseFromString(data_file)
    

    assert isinstance(data.state, str) and isinstance(data.action, int) and isinstance(data.reward, float) and isinstance(data.next_state, str) and isinstance(data.done, bool):
        
# 省略
# @pytest.mark.finished
# def test_model_load(state_size, action_size):
#     path = 'save/model.h5'
#     if os.path.exists(path):
#         actor = define_model(state_size, action_size)
#         assert actor.load_weights(path)

