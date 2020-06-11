import pytest
import torch
from torch.utils.data import DataLoader
from seisml.datasets import MarsInsight, mars_insight_transform
import os


class TestMarsInsight:

    def test_get_item(self):
        ds = MarsInsight(
            data_dir=os.path.expanduser('~/.seisml/mars/all_BH/prepared_12-3_oct-dec'),
            transform=mars_insight_transform()
        )

        dl = DataLoader(ds, batch_size=2, num_workers=1)
        sample = next(iter(dl))

        print('data length', len(ds))

        assert isinstance(sample[0], torch.Tensor), 'data output should be tensor'
        assert len(sample[0]) == 3, 'should have 3 channels'
        assert torch.mean(sample[0][0]).isclose(torch.Tensor([0]), atol=1e-5), 'mean should be removed'
        assert torch.mean(sample[0][1]).isclose(torch.Tensor([0]), atol=1e-5), 'mean should be removed'
        assert torch.mean(sample[0][2]).isclose(torch.Tensor([0]), atol=1e-5), 'mean should be removed'
