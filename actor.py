import os
import pickle
from argparse import ArgumentParser
from itertools import count
from multiprocessing import Process
from pathlib import Path

import numpy as np
import zmq

from common import init_components
from core import DataCollection
from utils import logger
from utils.cmdline import parse_cmdline_kwargs

parser = ArgumentParser()
parser.add_argument('--alg', type=str, help='The RL algorithm', required=True)
parser.add_argument('--env', type=str, help='The game environment', required=True)
parser.add_argument('--num_steps', type=float, help='The number of training steps', required=True)
parser.add_argument('--ip', type=str, help='IP address of learner server', required=True)
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--num_replicas', type=int, default=1, help='The number of actors')
parser.add_argument('--model', type=str, default=None, help='Training model')
parser.add_argument('--n_step', type=int, default=1, help='The number of sending data')
parser.add_argument('--log_path', type=str, default=None, help='Directory to save logging data')
parser.add_argument('--ckpt_path', type=str, default='./ckpt/', help='Directory to save model parameters')


def run_one_agent(index, args, unknown_args):
    from tensorflow.keras.backend import set_session
    import tensorflow.compat.v1 as tf

    # Set 'allow_growth'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    # Connect to learner
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.data_port}')

    env, agent = init_components(args, unknown_args)

    episode_rewards = [0.0]

    # Configure logging only in one process
    if index == 0:
        logger.configure(args.log_path)
    else:
        logger.configure(args.log_path, format_strs=[])

    data_collection = DataCollection(args.n_step)

    model_id = -1

    state = env.reset()
    for step in range(args.num_steps):
        # Do some updates
        agent.update_sampling(step, args.num_steps)

        # Sample action
        action, action_prob = agent.sample(state)
        next_state, reward, done, info = env.step(action)

        data = data_collection.push(state, action, action_prob, reward, next_state, done)
        if data is not None:
            socket.send(data)
            socket.recv()

        # Update weights
        new_weights = find_new_weights(model_id, args.ckpt_path)
        if new_weights is not None:
            agent.set_weights(new_weights)

        state = next_state
        episode_rewards[-1] += reward

        if done:
            num_episodes = len(episode_rewards)
            mean_10ep_reward = round(np.mean(episode_rewards[-10:]), 2)

            logger.record_tabular("steps", step)
            logger.record_tabular("episodes", num_episodes)
            logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
            logger.dump_tabular()

            state = env.reset()
            episode_rewards.append(0.0)


def run_weights_subscriber(args):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):
        weights = socket.recv()
        with open(args.ckpt_path / f'{model_id}.{args.alg}.{args.env}.ckpt', 'wb') as f:
            f.write(weights)


def find_new_weights(current_model_id: int, ckpt_path: Path):
    try:
        latest_file = max(ckpt_path.glob('*'), key=lambda p: p.stat().st_ctime)
    except ValueError:
        # No checkpoint file
        return None

    if int(latest_file.name.split('.')[0]) > current_model_id:
        with open(latest_file, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def main():
    # Parse input parameters
    parsed_args, unknown_args = parser.parse_known_args()
    parsed_args.num_steps = int(parsed_args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Create checkpoint path
    parsed_args.ckpt_path = Path(parsed_args.ckpt_path)
    i = 0
    while parsed_args.ckpt_path.exists():
        parsed_args.ckpt_path = parsed_args.ckpt_path.parent / (parsed_args.ckpt_path.name + str(i))
        i += 1
    else:
        parsed_args.ckpt_path.mkdir()

    # Run weights subscriber
    subscriber = Process(target=run_weights_subscriber, args=(parsed_args,))
    subscriber.start()

    if parsed_args.num_replicas > 1:
        agents = []
        for i in range(parsed_args.num_replicas):
            p = Process(target=run_one_agent, args=(i, parsed_args, unknown_args))
            p.start()
            os.system(f'taskset -p -c {i % os.cpu_count()} {p.pid}')  # For CPU affinity

            agents.append(p)

        for agent in agents:
            agent.join()
    else:
        run_one_agent(0, parsed_args, unknown_args)

    subscriber.join()


if __name__ == '__main__':
    main()
