import argparse

import numpy as np

from common import init_components
from utils.cmdline import parse_cmdline_kwargs

from tensorflow.keras.backend import set_session
import tensorflow.compat.v1 as tf

# Set 'allow_growth'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


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
            print("Average reward over last 10 trials:", average_reward, "last reward:", episode_rewards[-1])

            state = env.reset()

            episode_rewards.append(0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--num_steps', type=int, default=2000000)
    parser.add_argument('--model', type=str, default='accnn', help='Training model')
    parser.add_argument('--n_step', type=int, default=20, help='The number of sending data')

    args = parser.parse_args()

    parsed_args, unknown_args = parser.parse_known_args()
    parsed_args.num_steps = int(parsed_args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)
    unknown_args['verbose'] = False

    train(parsed_args, unknown_args)
