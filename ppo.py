import argparse

import numpy as np
from utils.cmdline import parse_cmdline_kwargs
from common import init_components


class DataCollection:
    """
    Send data when an episode is done or the number of collected data equals `size`
    """

    def __init__(self, size):
        self.buffer = []
        self.next_state = None
        self.done = None
        self.size = size

    def push(self, state, action, value, neglogp, reward, next_state, done):
        self.buffer.append([state, action, value, neglogp, reward])
        self.next_state = next_state
        self.done = done

        if self.done or len(self.buffer) == self.size:
            items = list(zip(*self.buffer))
            states = np.stack(items[0])
            actions, values, neglogps, rewards = [np.array(x).reshape(-1) for x in items[1:]]
            next_state = self.next_state

            data = [states, actions, values, neglogps, rewards, next_state, self.done]

            self.buffer = []
            self.next_state = None
            self.done = None

            return data
        else:
            return None


def train(args, unknown_args):

    env, agent1 = init_components(args, unknown_args)
    _, agent2 = init_components(args, unknown_args)
    agent1.set_weights(agent2.get_weights())

    episode_rewards = [0.0]

    data_collection = DataCollection(args.n_step)

    state = env.reset()

    for step in range(args.num_steps):

        ###  don't work
        action, value, neglogp = agent1.sample(state)
        ###

        ###  work
        # action, value, neglogp = agent2.sample(state)
        ###

        next_state, reward, done, info = env.step(action)

        data = data_collection.push(state, action, value, neglogp, reward, next_state, done)
        if data is not None:

            agent2.learn(*data, step)
            agent1.set_weights(agent2.get_weights())

        state = next_state
        episode_rewards[-1] += reward

        if done:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-10:]), 2)

            print(f'[Agent] Episode: {num_episodes}, Step: {step + 1}/{args.num_steps}, '
                  f'Mean Reward: {mean_100ep_reward}, Reward: {round(episode_rewards[-1], 2)}')

            state = env.reset()
            episode_rewards.append(0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--num_steps', type=int, default=40000)
    parser.add_argument('--model', type=str, default='acmlp', help='Training model')
    parser.add_argument('--n_step', type=int, default=5, help='The number of sending data')

    args = parser.parse_args()

    parsed_args, unknown_args = parser.parse_known_args()
    parsed_args.num_steps = int(parsed_args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    train(parsed_args, unknown_args)
