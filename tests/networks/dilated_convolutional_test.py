import pytest, os
from seisml.networks.dilated_convolutional import DilatedConvolutional
from seisml.datasets.triggered_earthquake import TriggeredEarthquake
from seisml.utility.download_data import download_sample_data
from seisml.metrics.loss import DeepClusteringLoss
import torch
from torch.utils.data import DataLoader
from torchsummary import summary


class TestDilatedConvolutional:

    def test_paper_configuration(self):
        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/sample_data/raw'),
            force_download=False,
            download=download_sample_data
        )

        dl = DataLoader(ds, batch_size=1, num_workers=1)

        model = DilatedConvolutional(embedding_size=10)
        data, label = next(iter(dl))
        summary(model, data.unsqueeze(1))

        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        assert num_params == 7440, 'number of params should match papers description'

    def test_single_pass_learn(self):
        embedding_size = 10

        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/sample_data/raw'),
            force_download=False,
            download=download_sample_data
        )

        dl = DataLoader(ds, batch_size=2, num_workers=1)

        model = DilatedConvolutional(embedding_size=embedding_size)
        data, label = next(iter(dl))
        params = filter(lambda p: p.requires_grad, model.parameters())
        opt = torch.optim.Adam(params, lr=0.01)
        l = DeepClusteringLoss()

        data = data.view(-1, 1, data.shape[-1])

        embedding_a = model(data)

        assert len(embedding_a[-1]) == embedding_size, 'output should match embedding size'

        _loss_a = l(embedding_a, label.float())
        _loss_a.backward()
        opt.step()

        embedding_b = model(data)
        _loss_b = l(embedding_b, label.float())

        assert _loss_b < _loss_a, 'the model should learn something'
