import datetime
import os
import pickle
import shutil
import time
from argparse import ArgumentParser
from itertools import count
from multiprocessing import Process, Array
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import zmq

from common import init_components
from core import DataCollection
from utils import logger
from utils.config import Config
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
parser.add_argument('--config_path', type=str, default=None, help='Directory to save config')
parser.add_argument('--ckpt_path', type=str, default=None, help='Directory to save model parameters')
parser.add_argument('--num_saved_ckpt', type=int, default=10, help='Number of recent checkpoint files to be saved')


def run_one_agent(index, args, unknown_args, actor_status):
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

    # Save config
    if index == 0:
        config = Config(parser, agent)
        config.save_config(args.config_path, "actor")
    else:
        pass

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
        new_weights, model_id = find_new_weights(model_id, args.ckpt_path)
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

    actor_status[index] = 1


def run_weights_subscriber(args, actor_status):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything

    for model_id in count(1):  # Starts from 1
        while True:
            try:
                weights = socket.recv(flags=zmq.NOBLOCK)

                # Weights received
                with open(args.ckpt_path / f'{model_id}.{args.alg}.{args.env}.ckpt', 'wb') as f:
                    f.write(weights)

                if model_id > args.num_saved_ckpt:
                    os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.{args.alg}.{args.env}.ckpt')
                break
            except zmq.Again:
                pass

            if all(actor_status):
                # All actors finished works
                return

            # For not cpu-intensive
            time.sleep(1)


def find_new_weights(current_model_id: int, ckpt_path: Path) -> Tuple[Any, int]:
    try:
        ckpt_files = sorted(os.listdir(ckpt_path.name), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    new_weights = pickle.load(f)
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass

        return new_weights, new_model_id
    else:
        return None, current_model_id


def main():
    # Parse input parameters
    parsed_args, unknown_args = parser.parse_known_args()
    parsed_args.num_steps = int(parsed_args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Remove existing log path
    if parsed_args.log_path is not None:
        parsed_args.log_path = Path(parsed_args.log_path)
        if parsed_args.log_path.exists():
            shutil.rmtree(parsed_args.log_path)

    # Create checkpoint path
    if parsed_args.ckpt_path is None:
        parsed_args.ckpt_path = 'ckpt-' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    parsed_args.ckpt_path = Path(parsed_args.ckpt_path)

    if parsed_args.ckpt_path.exists():
        shutil.rmtree(parsed_args.ckpt_path)
    parsed_args.ckpt_path.mkdir()

    # Running status of actors
    actor_status = Array('i', [0] * parsed_args.num_replicas)

    # Run weights subscriber
    subscriber = Process(target=run_weights_subscriber, args=(parsed_args, actor_status))
    subscriber.start()

    if parsed_args.num_replicas > 1:
        agents = []
        for i in range(parsed_args.num_replicas):
            p = Process(target=run_one_agent, args=(i, parsed_args, unknown_args, actor_status))
            p.start()
            os.system(f'taskset -p -c {i % os.cpu_count()} {p.pid}')  # For CPU affinity

            agents.append(p)

        for agent in agents:
            agent.join()
    else:
        run_one_agent(0, parsed_args, unknown_args, actor_status)

    subscriber.join()


if __name__ == '__main__':
    main()
