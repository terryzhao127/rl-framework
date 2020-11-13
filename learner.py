import zmq
import sys
from dqn.atari import AtariEnv
from dqn.cnn_model import CNNModel
from dqn.dqn_agent import DQNAgent
from dqn.protobuf.data import Data, bytes2arr


def main(argv):
    if argv[0] == 'test':
        dataloc = open('test/dataloc.out', 'rb')
        
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5000")

    env = AtariEnv('PongNoFrameskip-v4', 4)
    timesteps = 1000000

    dqn_agent = DQNAgent(
        CNNModel,
        env.get_observation_space(),
        env.get_action_space()
    )

    weight = b''
    weight_update = 0    
    for step in range(timesteps):

        if weight_update == 1:
            socket.send(weight)
        else:
            socket.send('no update')

        weight = b''

        if step == 100:
            message = socket.rev()
            with open('data.out', 'wb+') as f:
                f.write(message)
            continue
        
        data = Data()
        if argv[0] == 'test':
            data.ParseFromString(dataloc.readline().decode())
        else:
            data.ParseFromString(socket.recv())
        state, next_state = bytes2arr(data.state), bytes2arr(data.next_state)
        dqn_agent.memorize(state, data.action, data.reward, next_state, data.done)

        if step > dqn_agent.training_start:
            dqn_agent.learn()

            if step % dqn_agent.update_freq == 0:
                dqn_agent.update_target_model()
                weight_update = 1
                dqn_agent.save('save/model_{}.h5'.format(step))
            else:
                weight_update = 0

if __name__ == '__main__':
    main(sys.argv)
