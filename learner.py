import multiprocessing
import pickle
import subprocess
import time
from argparse import ArgumentParser
from itertools import count
from multiprocessing import Process

import horovod.tensorflow.keras as hvd
import tensorflow as tf
import zmq
from pyarrow import deserialize
from tensorflow.keras.backend import set_session

from common import init_components, load_yaml_config, save_yaml_config, create_experiment_dir
from core.mem_pool import MemPoolManager
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
    weights_socket = context.socket(zmq.PUB)
    weights_socket.bind(f'tcp://*:{args.param_port}')

    env, agent = init_components(args, unknown_args)

    # Save configuration file
    create_experiment_dir(args, 'LEARNER-')
    save_yaml_config(args.exp_path / 'config.yaml', args, 'learner', agent)

    # Record commit hash
    with open(args.exp_path / 'hash', 'w') as f:
        f.write(str(subprocess.run('git rev-parse HEAD'.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')))

    # Throughput-related data
    time_start = multiprocessing.Value('d', time.time())
    last_receiving_time = multiprocessing.Value('d', 0)
    last_consuming_time = multiprocessing.Value('d', 0)
    num_receiving_data = multiprocessing.Value('i', 0)
    num_consuming_data = multiprocessing.Value('i', 0)

    # Start memory pool in another process
    manager = MemPoolManager()
    manager.start()
    mem_pool = manager.MemPool(capacity=args.pool_size)
    Process(target=recv_data, args=(args.data_port, mem_pool, last_receiving_time, num_receiving_data)).start()

    # Print throughput statistics
    Process(target=print_throughput_fps,
            args=(last_receiving_time, last_consuming_time, num_receiving_data, num_consuming_data, time_start)).start()

    for step in count(1):
        # Do some updates
        agent.update_training(step, args.num_steps)

        if step % args.training_freq == 0 and len(mem_pool) >= args.batch_size:
            # Training
            agent.learn(mem_pool.sample(size=args.batch_size))
            num_consuming_data.value += args.batch_size
            last_consuming_time.value = time.time()

            # Sync weights to actor
            if hvd.rank() == 0:
                weights_socket.send(pickle.dumps(agent.get_weights()))


def recv_data(data_port, mem_pool, last_receiving_time, num_receiving_data):
    context = zmq.Context()
    data_socket = context.socket(zmq.REP)
    data_socket.bind(f'tcp://*:{data_port}')

    while True:
        # noinspection PyTypeChecker
        data: dict = deserialize(data_socket.recv())
        data_socket.send(b'200')
        mem_pool.push(data)

        num_receiving_data.value += len(data[list(data.keys())[0]])
        last_receiving_time.value = time.time()


def print_throughput_fps(last_receiving_time, last_consuming_time, num_receiving_data, num_consuming_data, time_start,
                         interval=10):
    """Print receiving and consuming periodically"""
    while True:
        time.sleep(interval)

        if last_receiving_time.value != 0 and last_consuming_time.value != 0:
            print(f'Receiving FPS: {num_receiving_data.value / (last_receiving_time.value - time_start.value):.2f}, '
                  f'Consuming FPS: {num_consuming_data.value / (last_consuming_time.value - time_start.value):.2f}')


if __name__ == '__main__':
    main()
