import pytest
import numpy as np
import torch
from seisml.datasets import triggered_tremor_split
from seisml.networks import ConvNet
from torchsummary import summary
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


class TestConvNet:

    def test_train(self):
        ds_train, ds_test = triggered_tremor_split(batch_size=1)

        model = ConvNet(
            input_shape=(1, 100000),
            num_layers=3,
            hidden_dims=(16,16,16),
            conv_kernel=2,
            pool_factor=10
        )
        optimizer = Adam(model.parameters(), lr=0.0001)
        loss = CrossEntropyLoss()

        summary(model, (1, 100000))

        device = torch.device('cpu')

        model.to(device)
        model.train()

        best_loss = 10000
        for i in range(3):
            history = []
            for batch in ds_train:
                data, label = batch

                data = data.unsqueeze(1)
                data = data.to(device)
                label = torch.argmax(label.squeeze(1), dim=1).to(device)

                optimizer.zero_grad()

                output = model(data)
                _loss = loss(output, label)
                _loss.backward()
                optimizer.step()

                history.append(_loss.item())

            epoch_best = np.mean(history)
            assert epoch_best < best_loss, 'loss should go down'
            best_loss = epoch_best
            history = []


