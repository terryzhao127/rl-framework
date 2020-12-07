from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from core import Agent


class PPOAgent(Agent):
    def __init__(self, model_cls, observation_space, action_space, config=None, gamma=0.98, lam=0.99, clip_range=0.1,
                 ent_coef=0.001, lr=0.0005, *args, **kwargs):
        # Default configurations
        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.lr = lr

        # Default model config
        if config is None:
            config = {}
        config['model'] = [{'model_id': 'policy_model'}]

        super(PPOAgent, self).__init__(model_cls, observation_space, action_space, config, *args, **kwargs)

        self.policy_model = self.model_instances[0]

        # Init
        input_shape = (None, observation_space) if isinstance(observation_space, int) else (None, *observation_space)
        self.states = tf.placeholder(tf.float32, shape=input_shape, name='states')
        self.actions_old = tf.placeholder(tf.float32, shape=(None,), name='actions_old')
        self.values_old = tf.placeholder(tf.float32, shape=(None,), name='values_old')
        self.neglogps_old = tf.placeholder(tf.float32, shape=(None,), name='neglogps_old')
        self.gaes = tf.placeholder(tf.float32, shape=(None,), name='advantage')
        self.q_values = tf.placeholder(tf.float32, shape=(None,), name='estimation')

        self.logits, value = self.policy_model.forward(self.states)
        self.action_distrs, self.values = tfp.distributions.Categorical(self.logits), tf.squeeze(value, axis=1)

        # Sample actions from the given distribution
        self.actions = self.action_distrs.sample()
        self.neglogp = -self.action_distrs.log_prob(self.actions)
        self.neglogp_new = -self.action_distrs.log_prob(self.actions_old)

        with tf.variable_scope('critic_loss'):
            self.values_cliped = self.values_old + \
                                 tf.clip_by_value(self.values - self.values_old, -self.clip_range, self.clip_range)
            critic_loss = tf.square(self.q_values - self.values)
            critic_loss_clipped = tf.square(self.q_values - self.values_cliped)
            self.critic_loss = tf.reduce_mean(tf.maximum(critic_loss, critic_loss_clipped))

        with tf.variable_scope('actor_loss'):
            ratio = tf.exp(self.neglogps_old - self.neglogp_new)
            actor_loss = self.gaes * ratio
            actor_loss_clipped = self.gaes * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            self.actor_loss = -tf.reduce_mean(tf.minimum(actor_loss, actor_loss_clipped))

        with tf.variable_scope('entropy_loss'):
            entropy = self.action_distrs.entropy()
            self.entropy_loss = tf.reduce_mean(entropy, axis=0)

        with tf.variable_scope('total_loss'):
            self.loss = self.critic_loss + self.actor_loss - self.entropy_loss * self.ent_coef

        # Calculate the gradients
        grads = tf.gradients(self.loss, self.policy_model.trainable_variables)
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(
            list(zip(grads, self.policy_model.trainable_variables))
        )

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def learn(self, states, actions, values, neglogps, rewards, next_state, done, step, *args, **kwargs) -> None:
        values = np.append(values, self.sample(next_state)[1] * (1 - done))

        gaes = rewards + self.gamma * values[1:] - values[:-1]
        for t in reversed(range(len(gaes) - 1)):
            gaes[t] = gaes[t] + self.gamma * self.lam * gaes[t + 1]

        q_values = np.zeros_like(rewards)
        cumulative = 0.0
        for i in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[i]
            q_values[i] = cumulative
        values = values[:-1]

        return self.sess.run([self.loss, self.train_op], feed_dict={self.states: states,
                                                                    self.actions_old: actions,
                                                                    self.values_old: values,
                                                                    self.neglogps_old: neglogps,
                                                                    self.gaes: gaes,
                                                                    self.q_values: q_values})

    def sample(self, state: Any, *args, **kwargs) -> Any:
        action, value, neg_logp = self.sess.run([self.actions, self.values, self.neglogp],
                                                feed_dict={self.states: state[np.newaxis]})
        return action[0], value[0], neg_logp[0]

    def policy(self, state: Any, *args, **kwargs) -> Any:
        logit = self.sess.run([self.logits], feed_dict={self.states: state[np.newaxis]})
        return np.argmax(logit[0])

    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        raise NotImplemented

    def set_weights(self, weights, *args, **kwargs):
        self.policy_model.set_weights(weights)

    def get_weights(self, *args, **kwargs):
        return self.policy_model.get_weights()

    def update_sampling(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass

    def update_training(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass

    def save(self, path, *args, **kwargs) -> None:
        self.policy_model.model.save(path)

    def load(self, path, *args, **kwargs) -> None:
        self.policy_model.model.load(path)
