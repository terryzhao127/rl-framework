from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

from core import Model


class CNNModel(Model):
    def __init__(self, observation_space, action_space, model_id='0', config=None, conv=None, fc=None,
                 *args, **kwargs):
        # Default configurations
        self.conv = [
            {'filters': 16, 'kernel_size': 8, 'strides': 4, 'activation': 'relu'},
            {'filters': 32, 'kernel_size': 4, 'strides': 2, 'activation': 'relu'},
        ] if conv is None else conv
        self.fc = [
            {'units': 256, 'activation': 'relu'},
            {'units': action_space, 'activation': 'linear'}
        ] if fc is None else fc

        # Define layers
        self.conv_layers = [Conv2D(**self.conv[0], input_shape=observation_space)]
        self.conv_layers += [Conv2D(**x) for x in self.conv[1:]]
        self.flatten = Flatten()
        self.dense_layers = [Dense(**x) for x in self.fc]

        self.model = None

        super(CNNModel, self).__init__(observation_space, action_space, model_id, config, *args, **kwargs)

    def build(self):
        self.model = Sequential()

        for conv_layer in self.conv_layers:
            self.model.add(conv_layer)
        self.model.add(self.flatten)
        for dense_layer in self.dense_layers:
            self.model.add(dense_layer)

    def set_weights(self, weights, *args, **kwargs):
        self.model.set_weights(weights)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights()

    def forward(self, states, *args, **kwargs):
        return self.model.predict(states)
