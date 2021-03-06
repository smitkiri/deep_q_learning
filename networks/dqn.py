from collections import OrderedDict
import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers=3, hidden_dim=256, **kwargs):
        """Deep Q-Network  PyTorch model.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Define the layers of the model such that
        # * there are `num_layers` nn.Linear modules / layers
        # * all activations except the last are ReLU activations
        input_dim = state_dim
        layers = []

        for idx in range(num_layers):
            # If this is the last layer, the output dim will be the action space
            if idx == num_layers - 1:
                output_dim = action_dim
            else:
                output_dim = hidden_dim

            # Add a linear layer
            layers.append((f"linear{idx + 1}", nn.Linear(input_dim, output_dim)))

            # Add a ReLU activation only if it is not the last layer
            if idx != num_layers - 1:
                layers.append((f"relu{idx + 1}", nn.ReLU()))

            input_dim = output_dim

        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, states) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param states: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of state-space.
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space.  This represents the Q values Q(s, .).
        """
        # Use the defined layers and activations to compute
        # the action-values tensor associated with the input states.
        return self.model(states)

    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.state_dim, self.action_dim),
            'kwargs': {
                'num_layers': self.num_layers,
                'hidden_dim': self.hidden_dim,
            },
            'state_dict': self.state_dict(),
        }
