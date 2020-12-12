from typing import Any

import numpy as np
from tensorflow.keras.optimizers import Adam

from core import Agent
from .replay_buffer import ReplayBuffer


class DQNAgent(Agent):
    def __init__(self, model_cls, observation_space, action_space, config=None, optimizer=None, batch_size=32,
                 epsilon=1, epsilon_min=0.01, gamma=0.99, buffer_size=5000, update_freq=1000, training_start=5000,
                 lr=0.001, exploration_fraction=0.1, epochs=1, verbose=True, *args, **kwargs):
        # Default configurations
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.update_freq = update_freq
        self.training_start = training_start
        self.exploration_fraction = exploration_fraction
        self.epochs = epochs
        self.verbose = verbose

        # Default model config
        if config is None:
            config = {}
        config['model'] = [
            {'model_id': 'policy_model'},
            {'model_id': 'target_model'}
        ]

        super(DQNAgent, self).__init__(model_cls, observation_space, action_space, config, *args, **kwargs)

        # Update target model
        self.policy_model = self.model_instances[0]
        self.target_model = self.model_instances[1]
        self.update_target_model()

        # Compile model
        if optimizer is None:
            optimizer = Adam(learning_rate=lr)
        self.policy_model.model.compile(loss='mean_squared_error', optimizer=optimizer)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size)

    def learn(self, states, actions, action_probs, rewards, next_state, done, step, *args, **kwargs) -> None:
        self.memory.add_batch(states, actions, rewards, next_state, done)

        if step > self.training_start:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            next_action = np.argmax(self.policy_model.forward(next_states), axis=-1)
            target = rewards + (1 - dones) * self.gamma * self.target_model.forward(next_states)[
                np.arange(self.batch_size), next_action]
            target_f = self.policy_model.forward(states)
            target_f[np.arange(self.batch_size), actions] = target
            self.policy_model.model.fit(states, target_f, epochs=self.epochs, verbose=self.verbose)

    def sample(self, state, *args, **kwargs):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_space), 1.0
        else:
            act_values = self.policy_model.forward(state[np.newaxis])
            return np.argmax(act_values[0]), 1.0

    def preprocess(self, state: Any, *args, **kwargs) -> Any:
        raise NotImplemented

    def set_weights(self, weights, *args, **kwargs):
        self.policy_model.set_weights(weights)
        self.update_target_model()

    def get_weights(self, *args, **kwargs):
        return self.policy_model.get_weights()

    def update_sampling(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        # Adjust Epsilon
        fraction = min(1.0, float(current_step) / (total_steps * self.exploration_fraction))
        self.epsilon = 1 + fraction * (self.epsilon_min - 1)

    def update_training(self, current_step: int, total_steps: int, *args, **kwargs) -> None:
        if current_step > self.training_start and current_step % self.update_freq == 0:
            self.update_target_model()

    def save(self, path, *args, **kwargs) -> None:
        self.policy_model.model.save(path)

    def load(self, path, *args, **kwargs) -> None:
        self.policy_model.model.load(path)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update_target_model(self):
        self.target_model.set_weights(self.policy_model.get_weights())
