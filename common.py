from typing import Tuple

from core import Agent, Env
from env import get_env, _get_gym_env_type
from models import MODEL
from algorithms import ALGORITHM


def init_components(args, unknown_args) -> Tuple[Env, Agent]:
    # Initialize environment
    env = get_env(args.env, **unknown_args)

    # Get model class
    if args.model is not None:
        model_cls = MODEL.get(args.model)
    else:
        env_type = _get_gym_env_type(args.env)
        if env_type == 'atari':
            model_cls = MODEL.get('qcnn')
        elif env_type == 'classic_control':
            model_cls = MODEL.get('qmlp')
        else:
            raise NotImplementedError(f'No default model for environment: {args.env!r})')

    # Initialize agent
    agent_cls = ALGORITHM.get(args.alg)
    agent = agent_cls(model_cls, env.get_observation_space(), env.get_action_space(),
                      **unknown_args)  # TODO: Add config interface

    return env, agent
