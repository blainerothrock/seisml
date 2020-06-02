import torch.nn as nn

class ConvNet(nn.Module):

    def __init__(self, input_size=100000, num_channels=1, num_layers=3, hidden_size=100, num_classes=2):
        super(ConvNet, self).__init__()

        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.CNN = nn.Sequential()
        self.CNN.add_module('cnn0', nn.Conv1d(1, 4, 2, stride=2))
        self.CNN.add_module('pool0', nn.AvgPool1d(2))
        self.CNN.add_module('bn0', nn.BatchNorm1d(4))
        self.CNN.add_module('a0', nn.ReLU())

        self.CNN.add_module('cnn1', nn.Conv1d(4, 8, 2, stride=2))
        self.CNN.add_module('pool1', nn.AvgPool1d(2))
        self.CNN.add_module('bn1', nn.BatchNorm1d(8))
        self.CNN.add_module('a1', nn.ReLU())

        self.CNN.add_module('cnn3', nn.Conv1d(8, 16, 2, stride=2))
        self.CNN.add_module('pool3', nn.AvgPool1d(2))
        self.CNN.add_module('bn3', nn.BatchNorm1d(16))
        self.CNN.add_module('a3', nn.ReLU())

        self.CNN.add_module('cnn4', nn.Conv1d(16, 32, 2, stride=2))
        self.CNN.add_module('pool4', nn.AvgPool1d(2))
        self.CNN.add_module('bn4', nn.BatchNorm1d(32))
        self.CNN.add_module('a4', nn.ReLU())

        # for l in range(num_layers):
        #     self.CNN.add_module('cnn{}'.format(l), nn.Conv1d(hidden_size, hidden_size, 2))
        #     self.CNN.add_module('pool{}'.format(l), nn.AvgPool1d(2))
        #     self.CNN.add_module('bn{}'.format(l), nn.BatchNorm1d(hidden_size))
        #     self.CNN.add_module('a{}'.format(l), nn.ReLU())


        self.fc = nn.Sequential()
        self.fc.add_module('fc0', nn.Linear(32 * int(input_size / (4 ** 4)), 32))
        self.fc.add_module('fc_a0', nn.ReLU())
        self.fc.add_module('fc1', nn.Linear(32, 16))
        self.fc.add_module('fc_a1', nn.ReLU())
        self.fc.add_module('out', nn.Linear(16, num_classes))

    def forward(self, X):
        X = self.CNN(X)
        X = X.flatten(start_dim=1)
        output = self.fc(X)

        return output