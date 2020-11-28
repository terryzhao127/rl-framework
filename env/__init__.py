import re
from collections import defaultdict
from typing import Type

import gym

from core.env import Env
from .atari import AtariEnv
from .classic_control import ClassicControlEnv

mapping = {
    'atari': AtariEnv,
    'classic_control': ClassicControlEnv
}


def _get_env_type(env_id):
    """Modified from https://github.com/openai/baselines/blob/master/baselines/run.py"""
    game_envs = defaultdict(set)

    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in game_envs.keys():
        env_type = env_id
        env_id = [g for g in game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)

    return env_type


def get_env_cls(env_id: str) -> Type[Env]:
    env_type = _get_env_type(env_id)

    if env_type is None or env_type not in mapping:
        ValueError(f'Unknown environment: {env_type}')

    return mapping[env_type]
