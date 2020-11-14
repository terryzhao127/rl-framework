#leatest.py
import os
import subprocess
import zmq
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
def test_data_train():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5000")

    subprocess.run('python learner.py test')

    data_test = open('test/datatest.out', 'wb')
    test_step = 5000
    for step in range(test_step):
        weights = socket.recv()
        if weights == 'no update':
                    pass
        else:
            print('New Model Recv!')
        socket.send(data_test.readline().decode())
    print('Test Finish!')
        
# 省略
# @pytest.mark.finished
# def test_model_load(state_size, action_size):
#     path = 'save/model.h5'
#     if os.path.exists(path):
#         actor = define_model(state_size, action_size)
#         assert actor.load_weights(path)

