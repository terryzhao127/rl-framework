import pickle
from argparse import ArgumentParser
from multiprocessing import Process

import time
import numpy as np
import zmq

from common import init_components
from core import Data, arr2bytes
from utils.cmdline import parse_cmdline_kwargs

parser = ArgumentParser()
parser.add_argument('--alg', type=str, help='The RL algorithm', required=True)
parser.add_argument('--env', type=str,
                    help='The game environment', required=True)
parser.add_argument('--num_steps', type=float,
                    help='The number of training steps', required=True)
parser.add_argument(
    '--ip', type=str, help='IP address of learner server', required=True)
parser.add_argument('--port', type=int, default=5000,
                    help='Learner server port')
parser.add_argument('--num_replicas', type=int, default=1,
                    help='The number of actors')
parser.add_argument('--model', type=str, default=None, help='Training model')


def run_one_agent(index, args, unknown_args):
    # Connect to learner
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.port}')

    env, agent = init_components(args, unknown_args)

    episode_rewards = [0.0]

    # test related
    start_time, last_round_time = time.time()
    testdir = 'test/testlogger'

    state = env.reset()
    for step in range(args.num_steps):

        round_time = time.time()
        # Do some updates
        agent.update_sampling(step, args.num_steps)

        # Sample action
        action = agent.sample(state)
        next_state, reward, done, info = env.step(action)

        # Send transition
        data = Data(
            state=arr2bytes(state),
            action=int(action),
            reward=reward,
            next_state=arr2bytes(next_state),
            done=done
        )
        socket.send(data.SerializeToString())

        # Update weights
        weights = socket.recv()
        if len(weights):
            agent.set_weights(pickle.loads(weights))

        state = next_state
        episode_rewards[-1] += reward

        if done:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 2)

            round_time = time.time()
            print(f'[Agent {index}] Episode: {num_episodes}, Step: {step + 1}/{args.num_steps}, '
                  f'Mean Reward: {mean_100ep_reward}, Round Time: {round_time - last_round_time}')
            last_round_time = round_time

            state = env.reset()
            episode_rewards.append(0.0)

        end_time = time.time()
        print(f'All Time Cost: {end_time - start_time}')


def main():
    # Parse input parameters
    parsed_args, unknown_args = parser.parse_known_args()
    parsed_args.num_steps = int(parsed_args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    if parsed_args.num_replicas > 1:
        agents = []
        for i in range(parsed_args.num_replicas):
            agents.append(Process(target=run_one_agent,
                                  args=(i, parsed_args, unknown_args)))
            agents[-1].start()

        for agent in agents:
            agent.join()
    else:
        run_one_agent(0, parsed_args, unknown_args)


if __name__ == '__main__':
    main()
