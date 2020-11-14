import numpy as np
import zmq

from dqn.atari import AtariEnv
from dqn.cnn_model import CNNModel
from dqn.dqn_agent import DQNAgent
from dqn.protobuf.data import Data, arr2bytes


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5000")

    env = AtariEnv('PongNoFrameskip-v4', 4)
    timesteps = 1000000

    dqn_agent = DQNAgent(
        CNNModel,
        env.get_observation_space(),
        env.get_action_space()
    )

    episode_rewards = [0.0]

    state = env.reset()
    data_test = open('./test/datatest', 'wb')
    for step in range(timesteps):
        if step>0:
            break
        weights = socket.recv()

        if weights.decode() == 'no update':
            pass
        else:
            dqn_agent.set_weights(weights)
            dqn_agent.save('save/model_{}.h5'.format(step))

        # Adjust Epsilon
        dqn_agent.adjust_epsilon(step, timesteps)

        # Sample action
        action = dqn_agent.sample(state)
        next_state, reward, done, info = env.step(action)

        # Send transition
        data = Data(
            state=arr2bytes(state),
            action=int(action),
            reward=reward,
            next_state=arr2bytes(next_state),
            done=done, epoch=step
        )
        socket.send(data.SerializeToString())
        data_test.write(data.SerializeToString())

        state = next_state
        episode_rewards[-1] += reward

        if done:
            num_episodes = len(episode_rewards)
            mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 2)

            print(f'Episode: {num_episodes}, Step: {step + 1}/{timesteps}, Mean Reward: {mean_100ep_reward}, '
                  f'Epsilon: {dqn_agent.epsilon:.3f}')

            state = env.reset()
            episode_rewards.append(0.0)
    data_test.close()

if __name__ == '__main__':
    main()
