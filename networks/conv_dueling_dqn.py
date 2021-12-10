from collections import OrderedDict
import torch.nn as nn


class DuelingDQNConvNet(nn.Module):
    def __init__(self, height, width, channels, action_dim, *, num_conv_layers=3, hidden_dim=128,
                 kernel_size=(3, 3), stride=(2, 2), **kwargs):
        """
        DQN PyTorch Model with Convolutional Neural Networks

        :param height:
        :param width:
        :param action_dim:
        :param num_conv_layers:
        :param hidden_dim:
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.action_dim = action_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_conv_layers = num_conv_layers
        self.hidden_dim = hidden_dim

        assert num_conv_layers > 0, "There should be at least 1 conv layer to process the image"

        conv_layers = []
        num_in_channels = channels
        num_out_channels = 8
        output_shape = (height, width)

        # Add convolution layers with Batch norm layers
        for idx in range(num_conv_layers):
            conv_layers.append((f"Conv{idx + 1}", nn.Conv2d(num_in_channels, num_out_channels,
                                                            kernel_size=kernel_size, stride=stride)))
            conv_layers.append((f"BatchNorm{idx + 1}", nn.BatchNorm2d(num_out_channels)))

            num_in_channels = num_out_channels
            num_out_channels = num_out_channels * 2

            output_shape = DuelingDQNConvNet.conv_output_shape(output_shape, kernel_size=kernel_size, stride=stride)

        # Linear input size
        liner_input_size = (output_shape[0] * output_shape[1] * num_out_channels) // 2

        self.conv_layers = nn.Sequential(OrderedDict(conv_layers))

        self.value_stream = nn.Sequential(
            nn.Linear(liner_input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(liner_input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

    @staticmethod
    def conv_output_shape(h_w, kernel_size=1, stride=(1, 1)):
        """
        Utility function for computing output of convolutions
        takes a tuple of (h,w) and returns a tuple of (h,w)
        """
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = (h_w[0] - (kernel_size[0] - 1) - 1) // stride[0] + 1
        w = (h_w[1] - (kernel_size[1] - 1) - 1) // stride[1] + 1
        return h, w

    def forward(self, x):
        img_output = self.conv_layers(x)
        img_output = img_output.view(img_output.size(0), -1)

        values = self.value_stream(img_output)
        advantages = self.advantage_stream(img_output)

        q_values = values + (advantages - advantages.mean())

        return q_values

    @classmethod
    def custom_load(cls, data):
        model = cls(*data['args'], **data['kwargs'])
        model.load_state_dict(data['state_dict'])
        return model

    def custom_dump(self):
        return {
            'args': (self.height, self.width, self.channels, self.action_dim),
            'kwargs': {
                'num_conv_layers': self.num_conv_layers,
                'hidden_dim': self.hidden_dim,
                'kernel_size': self.kernel_size,
                'stride': self.stride
            },
            'state_dict': self.state_dict(),
        }
