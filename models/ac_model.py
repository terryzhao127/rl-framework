import tensorflow.keras as tfk

from core import Model


class ACMLPModel(Model):
    def __init__(self, observation_space, action_space, model_id='0', config=None, *args, **kwargs):
        self.actor_model = tfk.Sequential()
        self.actor_model.add(tfk.layers.Dense(64, 'relu'))
        self.actor_model.add(tfk.layers.Dense(64, 'relu'))
        self.actor_model.add(tfk.layers.Dense(action_space, 'softmax'))

        self.critic_model = tfk.Sequential()
        self.critic_model.add(tfk.layers.Dense(64, 'relu'))
        self.critic_model.add(tfk.layers.Dense(64, 'relu'))
        self.critic_model.add(tfk.layers.Dense(1))

        self.model = None
        super(ACMLPModel, self).__init__(observation_space, action_space, model_id, config, *args, **kwargs)

    def build(self):
        input_x = tfk.Input(shape=(self.observation_space,))
        actor = self.actor_model(input_x)
        critic = self.critic_model(input_x)
        self.model = tfk.Model(inputs=input_x, outputs=(actor, critic))

    def set_weights(self, weights, *args, **kwargs):
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights()

    def forward(self, states, *args, **kwargs):
        return self.model(states)

    def predict(self, states):
        return self.model.predict(states)


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

    def predict(self, states):
        return self.model.predict(states)
