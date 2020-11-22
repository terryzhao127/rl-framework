from core.agent import Agent
from core.env import Env
from env.atari import AtariEnv
from .dqn.cnn_model import CNNModel
from .dqn.dqn_agent import DQNAgent

mapping = {
    'dqn': DQNAgent
}


def get_agent(alg_name: str, env: Env) -> Agent:
    model_cls = None
    if alg_name == 'dqn':
        if isinstance(env, AtariEnv):
            model_cls = CNNModel

    if model_cls is None:
        raise NotImplementedError(f"Unsupported environment for f{alg_name}: {env}")

    return mapping[alg_name](model_cls, env.get_observation_space(), env.get_action_space())
