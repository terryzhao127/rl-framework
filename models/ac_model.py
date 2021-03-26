from abc import abstractmethod, ABC
from typing import Any

import tensorflow as tf

import models.utils as utils
from models import model_registry
from models.tf_v1_model import TFV1Model

__all__ = ['ACModel', 'ACMLPModel', 'ACCNNModel']


class ACModel(TFV1Model, ABC):
    def __init__(self, observation_space, action_space, config=None, model_id='0', *args, **kwargs):
        with tf.variable_scope(model_id):
            self.x_ph = utils.placeholder(shape=observation_space)
            self.a_ph = utils.placeholder(dtype=tf.int32)

        # Output tensors
        self.pi = None
        self.logp = None
        self.logp_pi = None
        self.v = None

        super(ACModel, self).__init__(observation_space, action_space, config, model_id, scope=model_id,
                                      *args, **kwargs)

    def forward(self, states: Any, *args, **kwargs) -> Any:
        return self.sess.run([self.pi, self.v, self.logp_pi], feed_dict={self.x_ph: states})

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass


@model_registry.register('acmlp')
class ACMLPModel(ACModel):

    def build(self) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('pi'):
                logits = utils.mlp(self.x_ph, [64, 64, self.action_space], tf.tanh)
                self.pi, self.logp, self.logp_pi = utils.actor(logits, self.a_ph, self.action_space)

            with tf.variable_scope('v'):
                self.v = tf.squeeze(utils.mlp(self.x_ph, [64, 64, 1], tf.tanh), axis=1)


@model_registry.register('accnn')
class ACCNNModel(ACModel):

    def build(self) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('cnn_base'):
                layers = [{'filters': 32, 'kernel_size': 3, 'strides': 2, 'activation': 'relu'},
                          {'filters': 32, 'kernel_size': 3, 'strides': 2, 'activation': 'relu'},
                          {'filters': 32, 'kernel_size': 3, 'strides': 2, 'activation': 'relu'},
                          {'filters': 32, 'kernel_size': 3, 'strides': 2, 'activation': 'relu'}]
                feat = self.x_ph
                for layer in layers:
                    feat = tf.layers.conv2d(feat, **layer)
                feat = tf.layers.dense(tf.layers.flatten(feat), 512, activation='relu')

            with tf.variable_scope('pi'):
                logits = tf.layers.dense(feat, self.action_space)
                self.pi, self.logp, self.logp_pi = utils.actor(logits, self.a_ph, self.action_space)

            with tf.variable_scope('v'):
                self.v = tf.squeeze(tf.layers.dense(feat, 1), axis=1)
