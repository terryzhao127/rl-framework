from typing import Tuple

from agents import agent_registry
from core import Agent, Env
from env import get_env, _get_gym_env_type
from models import model_registry


def init_components(args, unknown_args) -> Tuple[Env, Agent]:
    # Initialize environment
    env = get_env(args.env, **unknown_args)

    # Get model class
    if args.model is not None:
        model_cls = model_registry.get(args.model)
    else:
        env_type = _get_gym_env_type(args.env)
        if env_type == 'atari':
            model_cls = model_registry.get('qcnn')
        elif env_type == 'classic_control':
            model_cls = model_registry.get('qmlp')
        else:
            raise NotImplementedError(f'No default model for environment: {args.env!r})')

    # Initialize agent
    agent_cls = agent_registry.get(args.alg)
    agent = agent_cls(model_cls, env.get_observation_space(), env.get_action_space(),
                      **unknown_args)  # TODO: Add config interface

    return env, agent
