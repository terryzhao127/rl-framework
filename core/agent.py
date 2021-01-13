from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type, Union, Dict

import numpy as np

from .model import Model
from .utils import get_config_params


class Agent(ABC):
    def __init__(self, model_cls: Type[Model], observation_space: Any, action_space: Any, config: dict = None,
                 *args, **kwargs) -> None:
        """
        1. Set configuration parameters (which appear after 'config') for agent and configurations for model
        if specified in 'config'
        2. Initialize model instances
        3. Build training part of computational graph
        :param model_cls: Model class that agent adopts
        :param observation_space: Env observation space
        :param action_space: Env action space
        :param config: Configurations for agent and models
        :param args: Positional configurations for agent only (ignored if specified in 'config')
        :param kwargs: Keyword configurations for agent only (ignored if specified in 'config')
        """
        self.model_cls = model_cls
        self.observation_space = observation_space
        self.action_space = action_space

        # Initialize instances of 'model_cls'
        self.model_instances = None

        if config is not None:
            self.load_config(config)
        else:
            self._init_model_instances(config)

    @abstractmethod
    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        """Preprocess the game state"""
        pass

    @abstractmethod
    def set_weights(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_weights(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def save(self, path: Path, *args, **kwargs) -> None:
        """Save the checkpoint file"""
        pass

    @abstractmethod
    def load(self, path: Path, *args, **kwargs) -> None:
        """Load the checkpoint file"""
        pass

    @abstractmethod
    def learn(self, episodes, *args, **kwargs) -> None:
        """Train the agent"""
        pass

    @abstractmethod
    def update_sampling(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        """Do some updates according to the process of sampling"""
        pass

    @abstractmethod
    def update_training(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        """Do some updates according to the process of training"""
        pass

    def export_config(self) -> dict:
        """Export dictionary as configurations"""
        param_dict = {p: str(getattr(self, p)) for p in get_config_params(Agent.__init__)}

        model_config = None
        if len(self.model_instances) == 1:
            model_config = self.model_instances[0].export_config()
        else:
            model_config = [x.export_config() for x in self.model_instances]
        param_dict.update({'model': model_config})

        return param_dict

    def load_config(self, config: dict) -> None:
        """Load dictionary as configurations and initialize model instances"""
        for key, val in config.items():
            if key in get_config_params(Agent.__init__):
                self.__dict__[key] = val

        self._init_model_instances(config)

    def predict(self, state: Any, *args, **kwargs) -> Any:
        """Get the action distribution at specific state"""
        return self.model_instances[0].forward(state, *args, **kwargs)

    def policy(self, state: Any, *args, **kwargs) -> Any:
        """Choose action during exploitation"""
        return np.argmax(self.predict(state, *args, **kwargs)[0])

    def sample(self, state: Any, *args, **kwargs) -> Dict:
        """Return action and other information(value, distribution et al) during exploration/sampling"""
        p = self.predict(state, *args, **kwargs)[0]
        return {'action': np.random.choice(len(p), p=p)}

    def _init_model_instances(self, config: Union[dict, None]) -> None:
        """Initialize model instances"""
        self.model_instances = []

        if config is not None and 'model' in config:
            model_config = config['model']

            if isinstance(model_config, list):
                for i, c in enumerate(model_config):
                    self.model_instances.append(self.model_cls(self.observation_space, self.action_space, **c))
            elif isinstance(model_config, dict):
                self.model_instances.append(
                    self.model_cls(self.observation_space, self.action_space, **model_config))
        else:
            self.model_instances.append(self.model_cls(self.observation_space, self.action_space))

    def format_data(self, data):
        """Format the data transferred to learner"""
        pass
