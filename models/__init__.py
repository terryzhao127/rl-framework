from typing import Type

from core.model import Model
from env import _get_gym_env_type
from .ac_model import ACMLPModel, ACCNNModel
from .cnn_model import CNNModel
from .mlp_model import MLPModel

mapping = {
    'cnn': CNNModel,
    'mlp': MLPModel,
    'acmlp': ACMLPModel,
    'accnn': ACCNNModel
}


def get_model_cls(name: str) -> Type[Model]:
    if name in mapping:
        return mapping[name]
    else:
        raise ValueError(f'Unknown model: {name}')


def get_default_model_cls(env_id):
    env_type = _get_gym_env_type(env_id)

    if env_type == 'atari':
        return CNNModel
    elif env_type == 'classic_control':
        return MLPModel
    else:
        raise NotImplementedError(f'No default model for environment: {env_id})')
