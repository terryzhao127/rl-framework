#leatest.py
import subprocess
import zmq
import pytest

def test_data_train():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5000")

    data_test = open('test/datatest', 'rb')
    message = data_test.read()
    test_step = 5000
    for step in range(test_step):
        if step == 0:
            subprocess.Popen('python learner.py test', shell=True)
        weights = socket.recv()
        if weights.decode() == 'no update':
            pass
        else:
            print('New Model Recv!')
        socket.send(message)
    print('Test Finish!')
