from algorithms.dqn.dqn_agent import DQNAgent
from models.ac_model import ACMLPModel
from algorithms import ppo
from models.mlp_model import MLPModel
from core import agent
from algorithms.ppo import PPOAgent
from env import get_env
import datetime
# env = get_env('CartPole-v1')
# agent = DQNAgent(MLPModel, env.get_observation_space(), env.get_action_space(),)
# print('*****************************')
# print(agent.export_config())


cur_time = datetime.datetime.now()
print(cur_time)