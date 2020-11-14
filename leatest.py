#leatest.py
import subprocess
import zmq

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

