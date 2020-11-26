#test_learner.py
import subprocess
import zmq

def test_data_train():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:4555")

    data_test = open('/workplace/rl-framework/test/datatest', 'rb')
    message = data_test.read()
    test_step = 5000
    for step in range(test_step):
        if step == 0:
            subp = subprocess.Popen('~/miniconda3/envs/framework/bin/python learner.py test', shell=True)
        weights = socket.recv()
        if weights.decode() == 'no update':
            pass
        else:
            print('New Model Recv!')
        socket.send(message)
    subp.kill()
    print('Test Finish!')
