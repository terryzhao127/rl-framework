import pickle
import subprocess
import time
from argparse import ArgumentParser
from itertools import count

import horovod.tensorflow.keras as hvd
import tensorflow as tf
import zmq
from pyarrow import deserialize
from tensorflow.keras.backend import set_session

from common import init_components, load_yaml_config, save_yaml_config, create_experiment_dir
from core.mem_pool import MemPool
from utils.cmdline import parse_cmdline_kwargs

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
set_session(tf.Session(config=config))
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
parser.add_argument('--env', type=str, default='CartPole-v1', help='The game environment')
parser.add_argument('--num_steps', type=float, default=2e5, help='The number of total training steps')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to receive training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server to publish model parameters')
parser.add_argument('--model', type=str, default='acmlp', help='Training model')
parser.add_argument('--pool_size', type=int, default=4000, help='The max length of data pool')
parser.add_argument('--training_freq', type=int, default=1, help='How many steps are between each training')
parser.add_argument('--batch_size', type=int, default=4000, help='The batch size for training')
parser.add_argument('--exp_path', type=str, default=None, help='Directory to save logging data and config file')
parser.add_argument('--config', type=str, default=None, help='The YAML configuration file')


def main():
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'learner')

    # Expose socket to actor(s)
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.bind(f'tcp://*:{args.data_port}')
    weights_socket = context.socket(zmq.PUB)
    weights_socket.bind(f'tcp://*:{args.param_port}')

    env, agent = init_components(args, unknown_args)

    # Save configuration file
    create_experiment_dir(args, 'LEARNER-')
    save_yaml_config(args.exp_path / 'config.yaml', args, 'learner', agent)

    # Record commit hash
    with open(args.exp_path / 'hash', 'w') as f:
        f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    mem_pool = MemPool(capacity=args.pool_size)

    time_start = time.time()
    num_consuming_data = 0
    num_receiving_data = 0
    last_consuming_time = None

    for step in count(1):
        # Do some updates
        agent.update_training(step, args.num_steps)

        # Receive data
        # noinspection PyTypeChecker
        data: dict = deserialize(data_socket.recv())
        data_socket.send(b'200')

        mem_pool.push(data)

        num_receiving_data += len(data[list(data.keys())[0]])
        last_receiving_time = time.time()

        if step % args.training_freq == 0 and len(mem_pool) >= args.batch_size:
            # Training
            agent.learn(mem_pool.sample(size=args.batch_size))
            num_consuming_data += args.batch_size
            last_consuming_time = time.time()

            # Sync weights to actor
            if hvd.rank() == 0:
                weights_socket.send(pickle.dumps(agent.get_weights()))

        # Logging receiving/consuming fps
        if last_consuming_time is not None:
            print(f'Receiving FPS: {num_receiving_data / (last_receiving_time - time_start)}, '
                  f'Consuming FPS: {num_consuming_data / (last_consuming_time - time_start)}')


if __name__ == '__main__':
    main()
