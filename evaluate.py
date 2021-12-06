import os
import pickle
from argparse import ArgumentParser
from common import init_components, load_yaml_config
from utils.cmdline import parse_cmdline_kwargs

parser = ArgumentParser()
parser.add_argument('--alg', type=str, default='ppo', help='The RL algorithm')
parser.add_argument('--env', type=str, default='CartPole-v1', help='The game environment')
parser.add_argument('--num_steps', type=float, default=2e5, help='The number of total training steps')
parser.add_argument('--ip', type=str, default='localhost', help='IP address of learner server')
parser.add_argument('--data_port', type=int, default=5000, help='Learner server port to send training data')
parser.add_argument('--param_port', type=int, default=5001, help='Learner server port to subscribe model parameters')
parser.add_argument('--model', type=str, default='acmlp', help='Training model')
parser.add_argument('--exp_path', type=str, default=None,
                    help='Directory to save logging data, model parameters and config file')
parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum length of trajectory')
parser.add_argument('--config', type=str, default=None, help='The YAML configuration file')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU to sample every action')
parser.add_argument('--ckpt_dir', type=str, default=None,
                    help='like "/u01/rl/", it offers the directory to find the better model')
parser.add_argument('--ckpt_file', type=str,
                    default="/u02/rl-framework/ACTOR-2021-11-30-10-38-39/ckpt/8884.dqn.CartPole-v1.ckpt",
                    help='select which ckpt file to restruct the model')

#default = None,


def test_model(args, unknown_args):
    # Initialize values
    model_rewards = {}
    model_lengths = {}

    # Initialize environment and agent instance
    env, agent = init_components(args, unknown_args)

    if args.ckpt_dir == None:
        if args.ckpt_file == None:
            print("Your input of ckpt_file is none,please try again!")
            assert args.ckpt_file
        else:
            model_id = int(args.ckpt_file.split("/")[-1].split(".")[0])
            model_rewards[model_id] = []
            print("model_id", model_id)
            for _ in range(5):
                model_lengths[model_id] = 0
                with open(args.ckpt_file, "rb") as f:
                    new_weights = pickle.load(f)
                    agent.set_weights(new_weights)

                state = env.reset()
                k_reward = 0
                while (1):
                    # Sample action
                    action, extra_data = agent.sample(state)
                    next_state, reward, done, info = env.step(action)

                    # Record current transition
                    k_reward += reward
                    model_lengths[model_id] += 1

                    state = next_state

                    is_terminal = done or model_lengths[model_id] >= args.max_episode_length > 0
                    if is_terminal:
                        # Reset environment
                        model_rewards[model_id].append(k_reward)
                        print("step_of_the_model_id_steps:", model_lengths[model_id])
                        break
            print("mean_reward_of_the_ckpt:", find_mean_reward(model_rewards[model_id]))
    else:
        ckpt_files = sorted(os.listdir(args.ckpt_dir), key=lambda p: int(p.split(".")[0]))
        for new_file in ckpt_files:
            print("\nnew_file:", new_file)
            model_id = int(new_file.split(".")[0])
            model_rewards[model_id] = []
            print("model_id", model_id)
            for _ in range(5):
                model_lengths[model_id] = 0
                with open(args.ckpt_dir + new_file, "rb") as f:
                    new_weights = pickle.load(f)
                    agent.set_weights(new_weights)

                state = env.reset()
                k_reward = 0
                while (1):
                    # Sample action
                    action, extra_data = agent.sample(state)
                    next_state, reward, done, info = env.step(action)

                    # Record current transition
                    # model_rewards[model_id].append(reward)
                    k_reward += reward
                    model_lengths[model_id] += 1
                    state = next_state

                    is_terminal = done or model_lengths[model_id] >= args.max_episode_length > 0
                    if is_terminal:
                        # Reset environment
                        model_rewards[model_id].append(k_reward)
                        print("step_of_the_model_id_steps:", model_lengths[model_id])
                        break

            print("mean_reward_of_the_ckpt:", find_mean_reward(model_rewards[model_id]))


# calculate the mean_reward but not used in this instance in true sense
def find_mean_reward(kwargs):
    lenth_model = len(kwargs)
    print("nums of model:", lenth_model)
    reward_all = 0
    for k in kwargs:
        reward_all += k
    return reward_all / lenth_model


if __name__ == '__main__':
    # Parse input parameters
    args, unknown_args = parser.parse_known_args()
    args.num_steps = int(args.num_steps)
    unknown_args = parse_cmdline_kwargs(unknown_args)

    # Load config file
    load_yaml_config(args, 'actor')

    # Disable GPU
    if not args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Run the model
    test_model(args, unknown_args)
