from abc import ABC, abstractmethod
from typing import Any

import tensorflow as tf

from .utils import get_config_params


class Model(ABC):
    def __init__(self, observation_space: Any, action_space: Any, model_id: str = '0', config: dict = None,
                 *args, **kwargs) -> None:
        """
        1. Set configuration parameters (which appear after 'config')
        2. Define layers and tensors
        3. Build model
        :param model_id: The identifier of the model
        :param config: Configurations of hyper-parameters
        :param args: Positional configurations (ignored if specified in 'config')
        :param kwargs: Keyword configurations (ignored if specified in 'config')
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.model_id = model_id
        self.config = config

        if config is not None:
            self.load_config(config)
        else:
            self.build()

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """Build the computational graph"""
        pass

    @abstractmethod
    def set_weights(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def get_weights(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def forward(self, states: Any, *args, **kwargs) -> Any:
        pass

    def export_config(self) -> dict:
        """Export dictionary as configurations"""
        config_params = get_config_params(Model.__init__)

        return {p: getattr(self, p) for p in config_params}

    def load_config(self, config: dict) -> None:
        """Load dictionary as configurations and build model"""
        for key, val in config.items():
            if key in get_config_params(Model.__init__):
                self.__dict__[key] = val

        self.build()

    def __call__(self, *args, **kwargs) -> Any:
        with tf.get_default_session() as sess:
            return self.forward(sess, *args, **kwargs)
