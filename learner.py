import pickle
from itertools import count
from argparse import ArgumentParser

import horovod.tensorflow.keras as hvd
import tensorflow as tf
import zmq
from tensorflow.keras import backend as K

from common import init_components
from core.data import parse_data
from utils.cmdline import parse_cmdline_kwargs

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

parser = ArgumentParser()
parser.add_argument('--alg', type=str, help='The RL algorithm', required=True)
parser.add_argument('--env', type=str, help='The game environment', required=True)
parser.add_argument('--num_steps', type=float, help='The number of training steps', required=True)
parser.add_argument('--port', type=int, default=5000, help='Learner server port')
parser.add_argument('--model', type=str, default=None, help='Training model')


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Expose socket to actor(s)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://*:{args.port}')

    env, agent = init_components(args, unknown_args)

    for step in count(1):
        # Do some updates
        agent.update_training(step, args.num_steps)

        # Receive data
        data = parse_data(socket.recv())

        # Training
        agent.learn(*data, step)

        # Sync weights to actor
        if hvd.rank() == 0:
            socket.send(pickle.dumps(agent.get_weights()))


if __name__ == '__main__':
    main()
