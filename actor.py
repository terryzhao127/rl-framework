import pickle

import numpy as np
import zmq

from algorithms.dqn.cnn_model import CNNModel
from algorithms.dqn.dqn_agent import DQNAgent
from core import Data, arr2bytes
from env.atari import AtariEnv


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://127.0.0.1:5000")

    env = AtariEnv('PongNoFrameskip-v4', 4)
    timesteps = 1000000

    dqn_agent = DQNAgent(
        CNNModel,
        env.get_observation_space(),
        env.get_action_space()
    )

    episode_rewards = [0.0]

    state = env.reset()
    for step in range(timesteps):
        # Adjust Epsilon
        dqn_agent.adjust_epsilon(step, timesteps)

        # Sample action
        action = dqn_agent.sample(state)
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
            dqn_agent.set_weights(pickle.loads(weights))

        episode_rewards[-1] += reward

        if done:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 2)

            print(f'Episode: {num_episodes}, Step: {step + 1}/{timesteps}, Mean Reward: {mean_100ep_reward}, '
                  f'Epsilon: {dqn_agent.epsilon:.3f}')

            state = env.reset()
            episode_rewards.append(0.0)


if __name__ == '__main__':
    main()
