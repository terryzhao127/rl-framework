from pathlib import Path
import tensorflow as tf
from typing import Any

from core import Model
from abc import ABC, abstractmethod


class TFV1Model(Model, ABC):
    def __init__(self, observation_space: Any, action_space: Any, model_id='0', config=None, session=None, scope=None,
                 *args, **kwargs):
        self.scope = scope

        # Initialize Tensorflow session
        if session is None:
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            session = tf.Session(config=tf_config)
        self.sess = session

        super(TFV1Model, self).__init__(observation_space, action_space, model_id, config, *args, **kwargs)

        # Build saver
        self.saver = tf.train.Saver(tf.trainable_variables())

    def set_weights(self, weights, *args, **kwargs) -> None:
        self.sess.run(
            [tf.assign(var, weight) for var, weight in zip(tf.trainable_variables(scope=self.scope), weights)]
        )

    def get_weights(self, *args, **kwargs) -> Any:
        return self.sess.run(tf.trainable_variables(self.scope))

    def save(self, path: Path, *args, **kwargs) -> None:
        self.saver.save(self.sess, str(path))

    def load(self, path: Path, *args, **kwargs) -> None:
        self.saver.restore(self.sess, str(path))

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, states: Any, *args, **kwargs) -> Any:
        pass
