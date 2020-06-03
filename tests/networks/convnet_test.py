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
        ds_train, ds_test = triggered_tremor_split()

        model = ConvNet(input_shape=(1, 100000))
        optimizer = Adam(model.parameters(), lr=0.001)
        loss = CrossEntropyLoss()

        summary(model, (1, 100000))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model.to(device)
        model.train()

        best_loss = 10000
        for i in range(1):
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

                history.append(_loss.detach().cpu().item())

            epoch_best = np.mean(history)
            assert epoch_best < best_loss, 'loss should go down'
            best_loss = epoch_best


