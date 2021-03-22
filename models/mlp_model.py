from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from core import Model


class MLPModel(Model):
    def __init__(self, observation_space, action_space, model_id='0', config=None, hidden=None, *args, **kwargs):
        # Default configurations
        self.hidden_layers = [
            {'units': 24, 'activation': 'relu'},
            {'units': 24, 'activation': 'relu'},
        ] if hidden is None else hidden

        # Define layers
        self.layers = [Dense(input_dim=observation_space, **self.hidden_layers[0])]
        self.layers += [Dense(**x) for x in self.hidden_layers[1:]]
        self.layers.append(Dense(action_space, activation='linear'))

        self.model = None

        super(MLPModel, self).__init__(observation_space, action_space, model_id, config, *args, **kwargs)

    def build(self):
        self.model = Sequential()
        for layer in self.layers:
            self.model.add(layer)

    def set_weights(self, weights, *args, **kwargs):
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights()

    def forward(self, states, *args, **kwargs):
        return self.model.predict(states)
