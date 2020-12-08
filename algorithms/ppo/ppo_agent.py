import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from core import Agent


class PPOAgent(Agent):
    def __init__(self, model_cls, observation_space, action_space, config=None, gamma=0.99, lr=1e-4, clip_range=0.2,
                 ent_coef=1e-3, epochs=10, verbose=True, *args, **kwargs):
        # Default configurations
        self.gamma = gamma
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
        self.model.model.compile(optimizer=optimizers.Adam(lr=self.lr), loss=[self._actor_loss, "mse"])

    def _actor_loss(self, act_adv_prob, y_pred):
        action, advantage, action_prob = tf.split(act_adv_prob, 3, axis=-1)

        act_mat = tf.equal(self.act_mat, tf.cast(action, tf.int64))
        act_mat = tf.cast(act_mat, tf.float32)
        action_prob = tf.tile(action_prob, [1, self.action_space])

        prob = act_mat * y_pred
        old_prob = act_mat * action_prob
        r = prob / (old_prob + 1e-10)

        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - self.clip_range,
                                                       max_value=1 + self.clip_range) * advantage) +
                       self.ent_coef * -(prob * K.log(prob + 1e-10)))

    def sample(self, state, *args, **kwargs):
        action_probs = np.squeeze(self.model.predict(state[np.newaxis])[0])
        action = np.random.choice(np.arange(self.action_space), p=action_probs)
        return action, action_probs[action]

    def learn(self, states, actions, action_probs, rewards, next_state, done, step, *args, **kwargs):
        q_values = np.zeros_like(rewards, dtype=np.float32)
        next_value = (1 - done) * self.model.predict(next_state[np.newaxis])[1].item()
        cumulative = next_value
        for t in reversed(range(len(rewards))):
            cumulative = cumulative * self.gamma + rewards[t]
            q_values[t] = cumulative
        q_values = (q_values - q_values.mean()) / (q_values.std() + 1e-10)

        pred_values = np.squeeze(self.model.predict(states)[1])
        # pred_values = np.append(pred_values, next_value)
        # deltas = rewards + self.gamma * pred_values[1:] - pred_values[:-1]
        #
        # for t in reversed(range(len(deltas) - 1)):
        #     deltas[t] += self.gamma * self.lam * deltas[t+1]
        #
        # advantage = (deltas - deltas.mean()) / (deltas.std() + 1e-6)
        advantage = q_values - pred_values

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