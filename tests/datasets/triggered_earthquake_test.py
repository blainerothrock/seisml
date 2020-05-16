import pytest
import os
import torch
import numpy as np
from seisml.utility.download_data import DownloadableData
from seisml.datasets import TriggeredEarthquake, DatasetMode
from torch.utils.data import DataLoader


class TestTriggeredEarthquake:

    def test_download_and_preproces(self):
        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value),
            force_download=False,
            downloadable_data=DownloadableData.SAMPLE_DATA
        )

        assert len(ds) > 0, 'files should exist'
        assert os.path.isdir(os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value)), 'data should exist'

    def test_get_item(self):
        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value),
            force_download=False,
            downloadable_data=DownloadableData.SAMPLE_DATA
        )

        dl = DataLoader(ds, batch_size=1, num_workers=1)
        sample = next(iter(dl))

        assert isinstance(sample[0], torch.Tensor), 'data ouput should be tensor'
        assert np.sum(sample[1].numpy()) == 1, 'one-hot encoding should contain exactly 1 class'

    def test_train_mode(self):
        data_path = os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value)
        accepted_labels = ['positive', 'negative']
        mode = DatasetMode.TRAIN
        testing_quakes = ['eq01']
        non_testing_quakes = ['eq02', 'eq03']
        train_ds = TriggeredEarthquake(
            data_dir=data_path,
            force_download=False,
            labels=accepted_labels,
            downloadable_data=DownloadableData.SAMPLE_DATA,
            mode=mode,
            testing_quakes=testing_quakes
        )
        for file in os.listdir(os.path.join(data_path, 'prepare_{}'.format(mode.value))):
            data = torch.load(os.path.join(data_path, 'prepare_{}'.format(mode.value), file))
            assert data['quake'] not in testing_quakes, 'should not contain testing quake'

        total_raw_count = 0
        for l in accepted_labels:
            for q in non_testing_quakes:
                count = len(os.listdir(os.path.join(data_path, 'raw', q, l)))
                total_raw_count += count

        ds_count = 0
        for _ in train_ds:
            ds_count += 1

        assert total_raw_count == ds_count, 'dataset should contain all examples'

    def test_test_mode(self):
        data_path = os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value)
        accepted_labels = ['positive', 'negative', 'chaos']
        testing_quakes = ['eq01']
        mode = DatasetMode.TEST
        train_ds = TriggeredEarthquake(
            data_dir=data_path,
            force_download=False,
            labels=accepted_labels,
            downloadable_data=DownloadableData.SAMPLE_DATA,
            mode=mode,
            testing_quakes=testing_quakes
        )
        for file in os.listdir(os.path.join(data_path, 'prepare_{}'.format(mode.value))):
            data = torch.load(os.path.join(data_path, 'prepare_{}'.format(mode.value), file))
            assert data['quake'] in testing_quakes, 'should only contain testing quake'

        total_raw_count = 0
        for l in accepted_labels:
            for q in testing_quakes:
                count = len(os.listdir(os.path.join(data_path, 'raw', q, l)))
                total_raw_count += count

        ds_count = 0
        for _ in train_ds:
            ds_count += 1

        assert total_raw_count == ds_count, 'dataset should contain all examples'

    def test_inference_mode(self):
        data_path = os.path.expanduser('~/.seisml/data/' + DownloadableData.SAMPLE_DATA.value)
        accepted_labels = ['positive', 'negative', 'chaos']
        testing_quakes = ['eq01', 'eq02']
        non_testing_quakes = ['eq03']
        mode = DatasetMode.INFERENCE
        train_ds = TriggeredEarthquake(
            data_dir=data_path,
            force_download=False,
            labels=accepted_labels,
            downloadable_data=DownloadableData.SAMPLE_DATA,
            mode=mode,
            testing_quakes=testing_quakes
        )

        for file in os.listdir(os.path.join(data_path, 'prepare_{}'.format(mode.value))):
            data = torch.load(os.path.join(data_path, 'prepare_{}'.format(mode.value), file))
            assert data['quake'] not in testing_quakes, 'should not contain testing quake'

        total_raw_count = 0
        for l in accepted_labels:
            for q in non_testing_quakes:
                count = len(os.listdir(os.path.join(data_path, 'raw', q, l)))
                total_raw_count += count

        ds_count = 0
        for _ in train_ds:
            ds_count += 1

        assert total_raw_count == ds_count, 'dataset should contain all examples'
