import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

from core import Agent
from core import ReplayBuffer


class PPOAgent(Agent):
    def __init__(self, model_cls, observation_space, action_space, config=None, gamma=0.99, lam=0.98, lr=1e-4,
                 buffer_size=0, clip_range=0.2, ent_coef=1e-2, epochs=10, verbose=True,
                 *args, **kwargs):
        # Default configurations
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.buffer_size = buffer_size
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

        self.memory = ReplayBuffer(self.buffer_size)

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
        action_probs, value = self.model.predict(state[np.newaxis])
        action_probs = np.squeeze(action_probs)
        action = np.random.choice(np.arange(self.action_space), p=np.squeeze(action_probs))
        return {'action': action, 'act_prob': action_probs[action], 'value': value.item()}

    def format_data(self, data):
        next_state, done = data['next_state'], data['done']

        values, reward = np.array(data['value']), np.array(data['reward'])
        values = np.append(values, (1 - done) * self.model.predict(next_state[np.newaxis])[1].item())
        deltas = reward + self.gamma * values[1:] - values[:-1]

        advantages = [0] * len(values)
        for t in reversed(range(len(values) - 1)):
            advantages[t] = self.gamma * self.lam * advantages[t+1] + deltas[t]

        data.update({'advantage': advantages[:-1]})

        del data['next_state']
        del data['done']
        del data['reward']

    def learn(self, episodes, *args, **kwargs):

        for episode in episodes:
            self.memory.extend(list(zip(*[episode[key] for key in [
                'state', 'action', 'act_prob', 'value', 'advantage']])))

        states, actions, act_probs, values, advantages = self.memory.all(clear=True)
        act_adv_prob = np.stack([actions, advantages, act_probs], axis=-1)
        self.model.model.fit([states], [act_adv_prob, values+advantages], epochs=self.epochs, verbose=self.verbose)

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
