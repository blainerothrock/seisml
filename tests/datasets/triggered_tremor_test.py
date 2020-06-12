import numpy as np
import os, torch
from seisml.utility.download_data import DownloadableData
from seisml.datasets import TriggeredTremor, triggered_tremor_split
from torch.utils.data import DataLoader


class TestTriggeredTremor:

    def test_download(self):
        dd = DownloadableData.TRIGGERED_TREMOR_100HZ

        ds = TriggeredTremor(
            data_dir=os.path.expanduser('~/.seisml/data/' + dd.value),
            force_download=False,
            downloadable_data=dd
        )

        assert len(ds) > 0, 'files should exist'
        assert os.path.isdir(
            os.path.expanduser('~/.seisml/data/' + dd.value)), 'data should exist'

    def test_get_item(self):
        dd = DownloadableData.TRIGGERED_TREMOR_SAMPLE

        ds = TriggeredTremor(
            data_dir=os.path.expanduser('~/.seisml/data/' + dd.value),
            force_download=False,
            downloadable_data=dd
        )

        dl = DataLoader(ds, batch_size=1, num_workers=1)
        sample = next(iter(dl))

        assert isinstance(sample[0], torch.Tensor), 'data ouput should be tensor'
        assert np.sum(sample[1].numpy()) == 1, 'one-hot encoding should contain exactly 1 class'

    def test_train_test_split(self):
        ds_train, ds_test = triggered_tremor_split(batch_size=1)

        for train_batch in ds_train:
            train_data, _ = train_batch
            for test_batch in ds_test:
                test_data, _ = test_batch

                assert not (train_data.numpy() == test_data.numpy()).all(), 'training sample should not match testing'

