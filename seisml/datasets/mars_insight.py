import os
import gin
import obspy
import torch
from torch.utils.data import Dataset
from seisml.core.transforms import DetrendFilter, DetrendType, ToTensor, Compose

def mars_insight_transform():
    transforms = [
        DetrendFilter(detrend_type=DetrendType.DEMEAN),
        ToTensor()
    ]

    return Compose(transforms)

# TODO: add download option
@gin.configurable()
class MarsInsight(Dataset):
    """
        unsupervised data set for Seismic data from the Mars Insight Lander. NOTE: data must be downloaded and prepared
        before running this dataset.
    :param data_dir: str: absolute path of the split data (indivisual mseed examples)
    :param transform: seis.core.Transform: transforms to be completed on each individial sample
    """

    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform

        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        stream = obspy.read(self.files[i])
        ts = []
        for trace in stream:
            d = {self.transform.source: trace}
            self.transform(d)
            ts.append(d[self.transform.output])

        return torch.stack(ts)