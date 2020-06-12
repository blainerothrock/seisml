import torch.nn as nn
import gin

@gin.configurable()
class ConvNet(nn.Module):

    def __init__(self,
                 input_shape=(1, 100000),
                 num_layers=5,
                 hidden_dims=(4, 8, 16, 32, 64),
                 conv_kernel=2,
                 stride=2,
                 pool_factor=2,
                 padding=0,
                 num_classes=2):

        super(ConvNet, self).__init__()

        self.input_shape = input_shape
        self.num_layers = num_layers
        self.num_classes = num_classes


        if not (isinstance(hidden_dims, tuple) and len(hidden_dims) == num_layers):
            raise ValueError('hidden dims must be a tuple of size num_layers')
        self.hidden_dims = hidden_dims

        if isinstance(conv_kernel, int):
            self.conv_kernels = [conv_kernel] * num_layers
        elif isinstance(conv_kernel, tuple) and len(conv_kernel) == num_layers:
            self.conv_kernels = conv_kernel
        else:
            raise ValueError('kernels is malformed, but be a int or list of ints of length num_layers')

        if isinstance(stride, int):
            self.strides = [stride] * num_layers
        elif isinstance(stride, tuple) and len(stride) == num_layers:
            self.strides = stride
        else:
            raise ValueError('strides is malformed, but be a int or list of ints of length num_layers')

        if isinstance(pool_factor, int):
            self.pool_factors = [pool_factor] * num_layers
        elif isinstance(pool_factor, tuple) and len(pool_factor) == num_layers:
            self.pool_factors = pool_factor
        else:
            raise ValueError('pool_kernel is malformed, but be a int or list of ints of length num_layers')

        if isinstance(padding, int):
            self.paddings = [padding] * num_layers
        elif isinstance(padding, tuple) and len(padding) == num_layers:
            self.paddings = padding
        else:
            raise ValueError('strides is malformed, but be a int or list of ints of length num_layers')

        self.CNN = nn.Sequential()

        in_channel = self.input_shape[0]
        in_dim = self.input_shape[-1]

        for l in range(num_layers):
            in_dim = int(in_dim / self.pool_factors[l])

            self.CNN.add_module('cnn{}'.format(l), nn.Conv1d(
                in_channel,
                self.hidden_dims[l],
                kernel_size=self.conv_kernels[l],
                stride=self.strides[l],
                padding=self.paddings[l]
            ))
            # self.CNN.add_module('pool{}'.format(l), nn.AvgPool1d(self.conv_kernels[l]))
            self.CNN.add_module('pool{}'.format(l), nn.AdaptiveAvgPool1d(in_dim))
            self.CNN.add_module('bn{}'.format(l), nn.BatchNorm1d(self.hidden_dims[l]))
            self.CNN.add_module('a{}'.format(l), nn.ReLU())

            in_channel = self.hidden_dims[l]


        self.fc = nn.Sequential()
        self.fc.add_module('fc0', nn.Linear(in_dim * self.hidden_dims[-1], 32))
        self.fc.add_module('fc_a0', nn.ReLU())
        self.fc.add_module('fc1', nn.Linear(32, 16))
        self.fc.add_module('fc_a1', nn.ReLU())
        self.fc.add_module('out', nn.Linear(16, num_classes))

    def forward(self, X):
        X = self.CNN(X)
        X = X.flatten(start_dim=1)
        output = self.fc(X)

        return output

    def _output_size(self, in_size, conv_kernel, stride, padding, pool_kernel):
        out = int(((in_size + 2*(padding) - conv_kernel) / stride) + 1)
        # out = out / 2
        return out