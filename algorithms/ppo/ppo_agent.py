import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from core import Agent


class PPOAgent(Agent):
    def __init__(self, model_cls, observation_space, action_space, config=None, gamma=0.99, lam=0.98, lr=1e-4, clip_range=0.2,
                 ent_coef=1e-2, epochs=10, verbose=True, *args, **kwargs):
        # Default configurations
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.verbose = verbose

        # Default model config
        if config is None:
            config = {}
        config['model'] = [{'model_id': 'policy_model'}]

        super(PPOAgent, self).__init__(model_cls, observation_space, action_space, config, *args, **kwargs)

        self.act_mat = np.arange(action_space)[np.newaxis]

        self.model = self.model_instances[0]
        self.model.model.compile(optimizer=optimizers.Adam(lr=self.lr), loss=[self._actor_loss, "huber_loss"])

    def _actor_loss(self, act_adv_prob, y_pred):
        action, advantage, action_prob = [tf.reshape(x, [-1]) for x in tf.split(act_adv_prob, 3, axis=-1)]
        action = tf.cast(action, tf.int32)
        index = tf.transpose(tf.stack([tf.range(tf.shape(action)[0]), action]))
        prob = tf.gather_nd(y_pred, index)

        r = prob / (action_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - self.clip_range,
                                                       max_value=1 + self.clip_range) * advantage) +
                       self.ent_coef * K.sum(- y_pred * K.log(y_pred + 1e-10), -1))

    def sample(self, state, *args, **kwargs):
        action_probs = np.squeeze(self.model.predict(state[np.newaxis])[0])
        action = np.random.choice(np.arange(self.action_space), p=action_probs)
        return action, action_probs[action]

    def learn(self, states, actions, action_probs, rewards, next_state, done, step, *args, **kwargs):
        next_value = (1 - done) * self.model.predict(next_state[np.newaxis])[1].item()
        pred_values = np.squeeze(self.model.predict(states)[1])
        pred_values = np.append(pred_values, next_value)

        deltas = rewards + self.gamma * pred_values[1:] - pred_values[:-1]

        gaes = np.zeros_like(pred_values)
        for t in reversed(range(len(deltas))):
            gaes[t] = self.gamma * self.lam * gaes[t + 1] + deltas[t]

        q_values = gaes + pred_values
        q_values = q_values[:-1]
        advantage = gaes[:-1]

        act_adv_prob = np.stack([actions, advantage, action_probs], axis=-1)

        self.model.model.fit([states], [act_adv_prob, q_values], epochs=self.epochs, verbose=self.verbose)

    def policy(self, state, *args, **kwargs):
        logit = self.model.predict(state[np.newaxis])[0]
        return np.argmax(logit[0])

    def preprocess(self, state, *args, **kwargs):
        raise NotImplementedError

    def set_weights(self, weights, *args, **kwargs):
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights()

    def update_sampling(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass

    def update_training(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        pass

    def save(self, path, *args, **kwargs) -> None:
        self.model.model.save(path)

    def load(self, path, *args, **kwargs) -> None:
        self.model.model.load(path)