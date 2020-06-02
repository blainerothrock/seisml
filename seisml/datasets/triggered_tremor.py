import os, shutil
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import numpy as np
import gin
import h5py
from  scipy import signal

from seisml.utility.download_data import download_and_verify, DownloadableData, downloadable_data_path


@gin.configurable()
def triggered_tremor_split(training_split=0.7, batch_size=128, shuffle=True):
    """
    Helper to get training and testing datasets

    Returns:
        training dataset and testing dataloaders
    """
    ds = TriggeredTremor()

    ds_size = len(ds)
    indices = list(range(ds_size))
    offset = int(np.floor(training_split * ds_size))

    if shuffle:
        np.random.shuffle(indices)

    train_indices, test_indices = indices[:offset], indices[offset+1:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dl = DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
    test_dl = DataLoader(ds, batch_size=batch_size, sampler=test_sampler)

    return train_dl, test_dl


@gin.configurable()
class TriggeredTremor(Dataset):
    """
    Data for Triggered Tremor observations. Data is loaded from a single HD5

    Parameters:
       data_dir (string): location to persist the data
       force_download (bool): use already persisted data
       downloadable_data (seisml.utility.DownloadableData): predefined dataset
       mode (string): 'train' or 'test' set
       training_split (float): percent of training data
       seed (int): random seed for splitting the data. should always be the same for training and testing datasets.
    """

    def __init__(
            self,
            data_dir=os.path.expanduser('~/.seisml/data/triggered_tremor_sample'),
            force_download=False,
            downloadable_data=DownloadableData.TRIGGERED_TREMOR_SAMPLE):

        # download data is it does not exist in path
        if not os.path.isdir(os.path.expanduser(data_dir)) or force_download:
            download_and_verify(downloadable_data, downloadable_data_path(downloadable_data))

        self.data_dir = os.path.expanduser(data_dir)

        filename = os.path.join(self.data_dir, '{}.h5'.format(downloadable_data))
        f = h5py.File(filename, 'r')

        # X = np.vstack([signal.resample(x, 10000) for x in f['X']])
        X = f['X']
        y = f['y']

        self.y = torch.nn.functional.one_hot(torch.Tensor(y).long(), 2)
        self.X = torch.Tensor(X).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
