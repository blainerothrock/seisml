import os, shutil
import torch
from torch.utils.data import Dataset
import numpy as np
import gin
import h5py

from seisml.utility.download_data import download_and_verify, DownloadableData, downloadable_data_path

def triggered_tremor_split():
    train = TriggeredTremor(mode='train')
    test = TriggeredTremor(mode='test')
    return train, test

@gin.configurable(blacklist=['mode'])
class TriggeredTremor(Dataset):
    """
    Data for Triggered Tremor observations. Data is loaded from a single HD5
    """

    def __init__(
            self,
            data_dir=os.path.expanduser('~/.seisml/data/triggered_tremor'),
            force_download=False,
            downloadable_data=DownloadableData.TRIGGERED_TREMOR,
            mode='train',
            training_split=0.7,
            seed=42):

        # download data is it does not exist in path
        if not os.path.isdir(os.path.expanduser(data_dir)) or force_download:
            download_and_verify(downloadable_data.value, downloadable_data_path(downloadable_data))

        self.data_dir = data_dir
        self.mode = mode
        self.training_split = training_split

        filename = os.path.join(data_dir, '{}.h5'.format(downloadable_data.value))
        f = h5py.File(filename, 'r')

        X = f.get('X').value
        y = f.get('y').value

        np.random.seed(seed)
        randomize = np.arange(len(y))
        np.random.shuffle(randomize)

        offset = int(len(y) * training_split)

        if mode == 'train':
            self.X = X[randomize][:offset]
            self.y = y[randomize][:offset]
        elif mode == 'test':
            self.X = X[randomize][offset:]
            self.y = y[randomize][offset:]

        # convert to tensors
        self.y = torch.nn.functional.one_hot(torch.Tensor(self.y).long(), 2)
        self.X = torch.Tensor(self.X).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
