import pickle
from argparse import ArgumentParser

import horovod.tensorflow.keras as hvd
import tensorflow as tf
import zmq
from tensorflow.keras import backend as K

from algorithms import get_agent
from common.cmd_utils import parse_cmdline_kwargs
from core.data import Data, bytes2arr
from env import get_env

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


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Expose socket to actor(s)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://*:{args.port}')


    # Initialize environment
    env = get_env(args.env, **unknown_args)

    # Initialize agent
    agent = get_agent(args.alg, env)

    for step in range(args.num_steps):
        # Do some updates
        agent.update_training(step, args.num_steps)

        # Receive data
        data = Data()
        data.ParseFromString(socket.recv())
        state, next_state = bytes2arr(data.state), bytes2arr(data.next_state)

        # Training
        agent.learn(state, data.action, data.reward, next_state, data.done, step)

        # Sync weights to actor
        if hvd.rank() == 0:
            socket.send(pickle.dumps(agent.get_weights()))


if __name__ == '__main__':
    main()
