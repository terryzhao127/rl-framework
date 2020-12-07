import argparse
from collections import deque
from itertools import count

import numpy as np

from algorithms.ppo.ppo_agent import PPOAgent
from env.classic_control import ClassicControlEnv
from models.ac_model import ACMLPModel


def train():
    # Init the env
    env = ClassicControlEnv(args.env)

    # Create the model
    agent = PPOAgent(ACMLPModel, env.get_observation_space(), env.get_action_space())

    ma_ep_rew = deque(maxlen=20)

    for epoch in range(args.epochs):
        states, actions, values, neglogps, rewards = [], [], [], [], []

        state, ep_rew = env.reset(), 0

        for _ in count(1):

            action, value, neglogp = agent.sample(state)

            states.append(state)
            actions.append(action)
            values.append(value)
            neglogps.append(neglogp)

            state, reward, done, info = env.step(action)

            rewards.append(reward)

            ep_rew += reward

            if done:
                values.append(0)
                ma_ep_rew.append(ep_rew)
                break

        agent.learn(states, actions, rewards, values, neglogps)
        print("Epoch: ", epoch, "MA_ep_rew: ", np.mean(ma_ep_rew))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--gamma', type=float, default=0.98, help='discount rate for future rewards')
    parser.add_argument('--lam', type=float, default=0.99, help='discount rate for gaes')
    parser.add_argument('--cliprange', type=float, default=0.1)
    parser.add_argument('--ent_coef', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    train()
