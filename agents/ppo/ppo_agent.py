from pathlib import Path
from typing import Tuple, Any, Dict, List

import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.train import AdamOptimizer

from agents import agent_registry
from core import Agent
from models import TFV1Model


@agent_registry.register('ppo')
class PPOAgent(Agent):
    def __init__(self, model_cls, observation_space, action_space, config=None, gamma=0.97, lam=0.98, pi_lr=3e-4,
                 vf_lr=1e-3, clip_range=0.2, ent_coef=1e-2, epochs=80, target_kl=0.01, *args, **kwargs):
        assert issubclass(model_cls, TFV1Model)

        # Default configurations
        self.gamma = gamma
        self.lam = lam
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.target_kl = target_kl

        # Default model config
        if config is None:
            config = {'model': [{'model_id': 'policy_model'}]}

        # Model related objects
        self.model = None
        self.pi_loss = None
        self.v_loss = None
        self.train_pi = None
        self.train_v = None
        self.approx_kl = None

        # Placeholder for training targets
        self.advantage_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.value_ph = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.act_prob_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

        super(PPOAgent, self).__init__(model_cls, observation_space, action_space, config, *args, **kwargs)

    def build(self) -> None:
        self.model = self.model_instances[-1]

        # Build losses and training operators
        ratio = tf.exp(self.model.logp - self.act_prob_ph)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.advantage_ph, clipped_ratio * self.advantage_ph))
        self.v_loss = tf.reduce_mean((self.value_ph - self.model.v) ** 2)
        self.approx_kl = tf.reduce_mean(self.act_prob_ph - self.model.logp)

        self.train_pi = AdamOptimizer(learning_rate=self.pi_lr).minimize(self.pi_loss)
        self.train_v = AdamOptimizer(learning_rate=self.vf_lr).minimize(self.v_loss)

        # Initialize variables
        self.model.sess.run(tf.global_variables_initializer())

    def sample(self, state: Any, *args, **kwargs) -> Tuple[Any, dict]:
        action, value, act_prob = self.model.forward(state[np.newaxis])
        return action[0], {'act_prob': act_prob.item(), 'value': value.item()}

    def learn(self, training_data, *args, **kwargs):
        for _ in range(self.epochs):
            kl, _ = self.model.sess.run([self.approx_kl, self.train_pi], feed_dict={
                self.model.x_ph: training_data['state'],
                self.model.a_ph: training_data['action'],
                self.advantage_ph: training_data['advantage'],
                self.value_ph: training_data['value'],
                self.act_prob_ph: training_data['act_prob']
            })
            if kl > 1.5 * self.target_kl:
                # Early stopping
                break

        for _ in range(self.epochs):
            self.model.sess.run(self.train_v, feed_dict={
                self.model.x_ph: training_data['state'],
                self.model.a_ph: training_data['action'],
                self.advantage_ph: training_data['advantage'],
                self.value_ph: training_data['value'],
                self.act_prob_ph: training_data['act_prob']
            })

    def prepare_training_data(self, trajectory: List[Tuple[Any, Any, Any, Any, bool, dict]]) -> Dict[str, np.ndarray]:
        states, actions, rewards = [np.array(i) for i in list(zip(*trajectory))[:3]]
        next_state = trajectory[-1][3]
        done = trajectory[-1][4]

        extra_data = [i[-1] for i in trajectory]
        values = np.array([x['value'] for x in extra_data])

        last_val = (1 - done) * self.model.forward(next_state[np.newaxis])[1].item()
        values = np.append(values, last_val)
        rewards = np.append(rewards, last_val)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        advantages = discount_cumulative_sum(deltas, self.gamma * self.lam)
        rewards_to_go = discount_cumulative_sum(rewards, self.gamma)[:-1]

        return {
            'state': states,
            'action': actions,
            'value': rewards_to_go,
            'act_prob': np.array([x['act_prob'] for x in extra_data]),
            'advantage': advantages
        }

    def post_process_training_data(self, training_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        advantage = training_data['advantage']

        mean = np.sum(advantage) / len(advantage)
        std = np.sqrt(np.sum((advantage - mean) ** 2) / len(advantage))
        training_data['advantage'] = (advantage - mean) / std

        return training_data

    def set_weights(self, weights, *args, **kwargs) -> None:
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs) -> Any:
        return self.model.get_weights()

    def save(self, path: Path, *args, **kwargs) -> None:
        self.model.save(path)

    def load(self, path: Path, *args, **kwargs) -> None:
        self.model.load(path)

    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        pass

    def update_sampling(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass

    def update_training(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass


def discount_cumulative_sum(x, discount):
    """
    Magic from RLLab for computing discounted cumulative sums of vectors.
    :param x: [x0, x1, x2]
    :param discount: Discount coefficient
    :return: [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """

    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
