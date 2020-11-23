import pickle
from argparse import ArgumentParser
from multiprocessing import Process

import numpy as np
import zmq

from algorithms import get_agent
from common.cmd_utils import parse_cmdline_kwargs
from core import Data, arr2bytes
from env import get_env

parser = ArgumentParser()
parser.add_argument('--alg', type=str, help='The RL algorithm', required=True)
parser.add_argument('--env', type=str, help='The game environment', required=True)
parser.add_argument('--num_steps', type=float, help='The number of training steps', required=True)
parser.add_argument('--ip', type=str, help='IP address of learner server', required=True)
parser.add_argument('--port', type=int, default=5000, help='Learner server port')
parser.add_argument('--num_replicas', type=int, default=1, help='The number of actors')


def run_one_agent(index, args, unknown_args):
    # Connect to learner
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f'tcp://{args.ip}:{args.port}')

    # Initialize environment
    env = get_env(args.env, **unknown_args)

    # Initialize agent
    agent = get_agent(args.alg, env)

    episode_rewards = [0.0]

    state = env.reset()
    for step in range(args.num_steps):
        # Do some updates
        agent.update_sampling(step, args.num_steps)

        # Sample action
        action = agent.sample(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

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

        episode_rewards[-1] += reward

        if done:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 2)

            print(f'[Agent {index}] Episode: {num_episodes}, Step: {step + 1}/{args.num_steps}, '
                  f'Mean Reward: {mean_100ep_reward}')

            state = env.reset()
            episode_rewards.append(0.0)


def main():
    # Parse input parameters
    parsed_args, unknown_args = parser.parse_known_args()
    parsed_args.num_steps = int(parsed_args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    agents = []
    for i in range(parsed_args.num_replicas):
        agents.append(Process(target=run_one_agent, args=(i, parsed_args, unknown_args)))
        agents[-1].start()

    for agent in agents:
        agent.join()


if __name__ == '__main__':
    main()
