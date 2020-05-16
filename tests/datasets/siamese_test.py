import pytest, os
from seisml.datasets import SiameseDataset, TriggeredEarthquake
from seisml.utility.download_data import DownloadableData
from torch.utils.data import DataLoader
import numpy as np


class TestSiamese:

    def test_dataset_wrapping(self):
        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value),
            force_download=False,
            downloadable_data=DownloadableData.SAMPLE_DATA
        )

        batch_size = 4

        ds = SiameseDataset(ds)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=1)
        data, label = next(iter(dl))

        data_batch = data.shape[0]
        label_batch = label.shape[0]
        data_channel = data.shape[1]
        label_channel = label.shape[1]

        assert data_batch == batch_size, 'data batch size should match'
        assert label_batch == batch_size, 'label batch size should match'
        assert data_channel == label_channel == 2, 'should have two channels'
        assert not np.allclose(data[0][0], data[0][1]), 'paired data points should not equal'
        assert not np.allclose(label[0][0], label[0][1]), 'paired labels should be different classifications'
