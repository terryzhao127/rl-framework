from typing import Type

from core.agent import Agent
from .dqn.dqn_agent import DQNAgent

mapping = {
    'dqn': DQNAgent
}


def get_agent_cls(alg_name: str) -> Type[Agent]:
    if alg_name in mapping:
        return mapping[alg_name]
    else:
        raise ValueError(f'Unknown algorithm: {alg_name}')
