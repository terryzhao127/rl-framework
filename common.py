import datetime
import time
from pathlib import Path
from typing import Tuple

import yaml

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


def save_yaml_config(config_path: Path, args, agent: Agent) -> None:
    with open(config_path, 'w') as f:
        args_config = {k: v for k, v in vars(args).items() if not k.endswith('path')}
        args_config['exp_path'] = str(args.exp_path)
        yaml.dump(args_config, f, sort_keys=False, indent=4)
        f.write('\n')
        yaml.dump(agent.export_config(), f, sort_keys=False, indent=4)


def create_experiment_dir(args, prefix: str) -> None:
    if args.exp_path is None:
        args.exp_path = prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.exp_path = Path(args.exp_path)

    if args.exp_path.exists():
        raise FileExistsError(f'Experiment directory {str(args.exp_path)!r} already exists')

    args.exp_path.mkdir()
