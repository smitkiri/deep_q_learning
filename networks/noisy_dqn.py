from collections import OrderedDict
import math

import torch.nn.functional as F
import torch.autograd as autograd
import torch.nn as nn
import torch

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if torch.cuda.is_available() \
    else autograd.Variable(*args, **kwargs)


class NoisyLinear(nn.Module):
    """Noisy linear layer"""
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQN(nn.Module):
    def __init__(self, state_dim, action_dim, *, num_layers=2, hidden_dim=128, **kwargs):
        """Noisy Deep Q-Network  PyTorch model.

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
        layers = [("linear1", nn.Linear(state_dim, hidden_dim))]
        input_dim = hidden_dim

        for idx in range(num_layers):
            # If this is the last layer, the output dim will be the action space
            if idx == num_layers - 1:
                output_dim = action_dim
            else:
                output_dim = hidden_dim

            # Add a linear layer
            layers.append((f"noisy_linear{idx + 1}", NoisyLinear(input_dim, output_dim)))

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

    def reset_noise(self):
        """Resets noise in the NoisyLinear layers"""
        for idx in range(1, self.num_layers + 1):
            noisy_layer = self.model.get_submodule(f"noisy_linear{idx}")
            noisy_layer.reset_noise()

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
