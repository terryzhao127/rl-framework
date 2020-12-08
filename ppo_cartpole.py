import argparse

import numpy as np

from common import init_components
from utils.cmdline import parse_cmdline_kwargs


def train(args, unknown_args):
    env, agent = init_components(args, unknown_args)

    episode_rewards = [0.0]

    state = env.reset()

    states, actions, action_probs, rewards = [], [], [], []

    for step in range(args.num_steps):
        action, action_prob = agent.sample(state=state)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        action_probs.append(action_prob)
        rewards.append(reward)

        state = next_state
        episode_rewards[-1] += reward

        if done or len(states) == args.n_step:
            states, actions, action_probs, rewards = np.array(states), np.array(actions), np.array(
                action_probs), np.array(rewards)
            agent.learn(states, actions, action_probs, rewards, next_state, done, step)
            states, actions, action_probs, rewards = [], [], [], []

        if done:
            average_reward = np.mean(np.array(episode_rewards)[-10:])
            print("Average reward over last 100 trials: ", average_reward)

            state = env.reset()

            episode_rewards.append(0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--num_steps', type=int, default=40000)
    parser.add_argument('--model', type=str, default='acmlp', help='Training model')
    parser.add_argument('--n_step', type=int, default=10, help='The number of sending data')

    args = parser.parse_args()

    parsed_args, unknown_args = parser.parse_known_args()
    parsed_args.num_steps = int(parsed_args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    train(parsed_args, unknown_args)