from typing import Tuple

from algorithms import get_agent_cls
from core import Agent, Env
from env import get_env_cls
from models import get_default_model_cls, get_model_cls


def init_components(args, unknown_args) -> Tuple[Env, Agent]:
    # Initialize environment
    env_cls = get_env_cls(args.env)
    env = env_cls(args.env, **unknown_args)

    # Get model class
    if args.model is not None:
        model_cls = get_model_cls(args.model)
    else:
        model_cls = get_default_model_cls(args.env)

    # Initialize agent
    agent_cls = get_agent_cls(args.alg)
    agent = agent_cls(model_cls, env.get_observation_space(), env.get_action_space(),
                      **unknown_args)  # TODO: Add config interface

    return env, agent
