from pathlib import Path
import tensorflow.keras as tfk

import tensorflow as tf
from typing import Any
from core import Model

from models.tf_v1_model import TFV1Model


class ACMLPModel(TFV1Model):
    def __init__(self, observation_space, action_space, model_id='0', config=None, *args, **kwargs):
        # Input placeholders
        with tf.variable_scope(model_id):
            self.x_ph = tf.placeholder(dtype=tf.float32, shape=(None, observation_space))
            self.a_ph = tf.placeholder(dtype=tf.int32, shape=(None,))

        # Output tensors
        self.pi = None
        self.logp = None
        self.logp_pi = None
        self.v = None

        super(ACMLPModel, self).__init__(observation_space, action_space, model_id, config, scope=model_id, *args,
                                         **kwargs)

    def build(self) -> None:
        with tf.variable_scope(self.scope):
            with tf.variable_scope('pi'):
                act_dim = self.action_space
                logits = mlp(self.x_ph, [64, 64, act_dim], tf.tanh, None)
                logp_all = tf.nn.log_softmax(logits)
                self.pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
                self.logp = tf.reduce_sum(tf.one_hot(self.a_ph, depth=act_dim) * logp_all, axis=1)
                self.logp_pi = tf.reduce_sum(tf.one_hot(self.pi, depth=act_dim) * logp_all, axis=1)

            with tf.variable_scope('v'):
                self.v = tf.squeeze(mlp(self.x_ph, [64, 64, 1], tf.tanh, None), axis=1)

    def forward(self, states: Any, *args, **kwargs) -> Any:
        return self.sess.run([self.pi, self.v, self.logp_pi], feed_dict={self.x_ph: states})


class ACCNNModel(Model):
    def __init__(self, observation_space, action_space, model_id='0', config=None, *args, **kwargs):
        self.base_model = tfk.Sequential()
        self.base_model.add(tfk.layers.Conv2D(32, 3, 2, activation='relu'))
        self.base_model.add(tfk.layers.Conv2D(32, 3, 2, activation='relu'))
        self.base_model.add(tfk.layers.Conv2D(32, 3, 2, activation='relu'))
        self.base_model.add(tfk.layers.Conv2D(32, 3, 2, activation='relu'))
        self.base_model.add(tfk.layers.Flatten())
        self.base_model.add(tfk.layers.Dense(512, 'relu'))

        self.actor_model = tfk.Sequential()
        self.actor_model.add(tfk.layers.Dense(action_space, 'softmax'))

        self.critic_model = tfk.Sequential()
        self.critic_model.add(tfk.layers.Dense(1))

        self.model = None
        super(ACCNNModel, self).__init__(observation_space, action_space, model_id, config, *args, **kwargs)

    def build(self):
        input_x = tfk.Input(shape=(*self.observation_space,))
        feat = self.base_model(input_x)
        actor = self.actor_model(feat)
        critic = self.critic_model(feat)
        self.model = tfk.Model(inputs=input_x, outputs=(actor, critic))

    def set_weights(self, weights, *args, **kwargs):
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights()

    def forward(self, states, *args, **kwargs):
        return self.model(states)

    def save(self, path: Path, *args, **kwargs) -> None:
        self.base_model.save_weights(path / 'base_model')
        self.actor_model.save_weights(path / 'actor_model')
        self.critic_model.save_weights(path / 'critic_model')

    def load(self, path: Path, *args, **kwargs) -> None:
        self.base_model.load_weights(path / 'base_model')
        self.actor_model.load_weights(path / 'actor_model')
        self.critic_model.load_weights(path / 'critic_model')


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
