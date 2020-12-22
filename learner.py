import pickle
from argparse import ArgumentParser
from collections import deque, defaultdict
from itertools import count

import horovod.tensorflow.keras as hvd
import numpy as np
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
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to receive training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server to publish model parameters')
parser.add_argument('--model', type=str, default=None, help='Training model')
parser.add_argument('--buffer_maxlen', type=int, default=100, help='The max length of training data buffer')
parser.add_argument('--training_freq', type=int, default=100, help='How many steps are between each training')
parser.add_argument('--training_start', type=int, default=5000, help='The start step of training')


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Expose socket to actor(s)
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.bind(f'tcp://*:{args.data_port}')
    weights_socket = context.socket(zmq.PUB)
    weights_socket.bind(f'tcp://*:{args.param_port}')

    env, agent = init_components(args, unknown_args)

    for step in count(1):
        # Do some updates
        agent.update_training(step, args.num_steps)

        # Receive data
        data = parse_data(data_socket.recv())
        data_socket.send(b'200')
        agent.add(*data)

        if step > args.training_start and step % args.training_freq == 0:
            # Training
            agent.learn(step)

            # Sync weights to actor
            if hvd.rank() == 0:
                weights_socket.send(pickle.dumps(agent.get_weights()))


if __name__ == '__main__':
    main()
