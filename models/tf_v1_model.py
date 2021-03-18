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

        self.build_assign()
        # Build saver
        self.saver = tf.train.Saver(tf.trainable_variables())

    def build_assign(self):
        self._weight_ph, self._to_assign = dict(), dict()
        variables = tf.trainable_variables(self.scope)
        for var in variables:
            self._weight_ph[var.name] = tf.placeholder(var.value().dtype, var.get_shape().as_list())
            self._to_assign[var.name] = var.assign(self._weight_ph[var.name])
        self._nodes = list(self._to_assign.values())

    def set_weights(self, weights, *args, **kwargs) -> None:
        feed_dict = {self._weight_ph[var.name]: weight
                     for (var, weight) in zip(tf.trainable_variables(scope=self.scope), weights)}

        self.sess.run(self._nodes, feed_dict=feed_dict)

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
