import torch
from torch import nn
import numpy as np

torch.set_default_dtype(torch.float64)


def set_random_state(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class PINN(nn.Module):

    def __init__(self, no_of_h_layers, no_of_neurons, input_dim=1,
                 seed=1, mean=0.0, std=1.0):
        super().__init__()
        set_random_state(seed)
        # parameters for normalizing the input to network
        # This has to come from a dataset and should be user-specified
        self.mu = mean
        self.sigma = std
        # Constructing the layers
        layers = nn.ModuleList()
        # Input layer
        layer0 = nn.Linear(input_dim, no_of_neurons)
        layers.extend([layer0, nn.Tanh()])
        for i in range(no_of_h_layers):
            layers.append(nn.Linear(no_of_neurons, no_of_neurons))
            layers.append(nn.Tanh())
        # Output layer
        final_layer = nn.Linear(no_of_neurons, 1)
        layers.append(final_layer)
        self.layers = layers

        # Initialize the layer weights - Applies it recursively
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            layer_shape = module.weight.shape  # (n_out_features, no_in_features)
            in_dim = layer_shape[1]
            std = 1. / np.sqrt(in_dim)
            module.weight.data.normal_(mean=0.0, std=std)

            if module.bias is not None:
                module.bias.data.normal_(mean=0.0, std=1.0)

    def normalize(self, x):
        x = (x - self.mu) / self.sigma
        return x

    def forward(self, x):
        """Normalizes input and then feeds it to network."""
        x = self.normalize(x)
        for layer in self.layers:
            x = layer(x)
        return x
