import pytest
import os
import torch
import numpy as np
from seisml.utility.download_data import download_sample_data
from seisml.datasets.triggered_earthquake import TriggeredEarthquake
from torch.utils.data import DataLoader


class TestTriggeredEarthquake:

    def test_download_and_preproces(self):
        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/sample_data/raw'),
            force_download=False,
            download=download_sample_data
        )

        assert len(ds) > 0, 'files should exist'
        assert os.path.isdir(os.path.expanduser('~/.seisml/data/sample_data/')), 'data should exist'

    def test_get_item(self):
        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/sample_data/raw'),
            force_download=False,
            download=download_sample_data
        )

        dl = DataLoader(ds, batch_size=1, num_workers=1)
        sample = next(iter(dl))

        assert isinstance(sample[0], torch.Tensor), 'data ouput should be tensor'
        assert np.sum(sample[1].numpy()) == 1, 'one-hot encoding should contain exactly 1 class'
