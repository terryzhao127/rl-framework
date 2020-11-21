import zmq
import pickle

from algorithms.dqn.atari import AtariEnv
from algorithms.dqn.cnn_model import CNNModel
from algorithms.dqn.dqn_agent import DQNAgent
from algorithms.dqn.protobuf.data import Data, bytes2arr

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
import horovod.tensorflow.keras as hvd


# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5000")

    env = AtariEnv('PongNoFrameskip-v4', 4)
    timesteps = 1000000

    dqn_agent = DQNAgent(
        CNNModel,
        env.get_observation_space(),
        env.get_action_space(),
        hvd.DistributedOptimizer(RMSprop(learning_rate=0.0001))
    )

    weight = b''
    for step in range(timesteps):

        socket.send(weight)
        weight = b''

        data = Data()
        data.ParseFromString(socket.recv())
        state, next_state = bytes2arr(data.state), bytes2arr(data.next_state)
        dqn_agent.memorize(state, data.action, data.reward, next_state, data.done)

        if step > dqn_agent.training_start:
            dqn_agent.learn(callbacks=callbacks)

            if step % dqn_agent.update_freq == 0:
                dqn_agent.update_target_model()
                if hvd.rank() == 0:
                    weight = pickle.dumps(dqn_agent.get_weights())


if __name__ == '__main__':
    main()
