import os, random
from torch.utils.data import Dataset
import numpy as np
import obspy
from multiprocessing import cpu_count
from seisml.utility.download_data import download_triggered_earthquake_data
from seisml.utility.utils import parallel_process, save_file
from seisml.core.transforms import Resample, \
    ButterworthPassFilter, FilterType, Compose, \
    ToTensor, Augment, AugmentationType, TargetLength


def triggered_earthquake_transform(
        sampling_rate=20.0,
        max_freq=2.0,
        min_freq=8.0,
        corner=2,
        aug_types=[AugmentationType.AMPLITUDE, AugmentationType.NOISE],
        aug_prob=0.5,
        target_length=20000):

    transforms = [
        Resample(sampling_rate=sampling_rate),
        ButterworthPassFilter(
            filter_type=FilterType.BANDPASS,
            min_freq=max_freq,
            max_freq=min_freq,
            corners=corner,
            zerophase=True
        ),
        Augment(augmentation_types=aug_types, probability=aug_prob),
        TargetLength(target_length=target_length, random_offset=True),
        ToTensor()
    ]

    return Compose(transforms, source='t', inplace=True)


class TriggeredEarthquake(Dataset):
    """
    Dataset for Triggered Earthquakes as refferenced in Automating the Detection of Dynamically Triggered Earthquakes
    via a Deep Metric Learning Algorithm

    data file structure:
        data_dir/raw/
            earthquake1/
                positive/
                    file1.sac
                    file2.sac
                    ...
                negative/
                    file1.sac
                    file2.sac
                    ...
                more_optional_labels_like_chaos/
                    more_files.sac
                    ...
            earthquake2/
                positive/
                    file1.sac
                    file2.sac
                    ...
                negative/
                    file1.sac
                    file2.sac
                    ...
                more_optional_labels_like_chaos/
                    more_files.sac
                    ...
            ...


    Args:
        data_dir: (str): folder where raw triggered earthquake data exists
        force_download: (Bool): [optional] if data already exists, re-download it, default: False
        download: (function): [optional] download method, default: download_triggered_earthquake_data
        labels: (list): labels to use from the directory structure, default: ['positive', 'negative']
    """

    def __init__(
            self,
            data_dir=os.path.expanduser('~/.seisml/data/triggered_earthquakes'),
            force_download=False,
            download=download_triggered_earthquake_data,
            labels=['positive', 'negative'],
            transform=triggered_earthquake_transform()):

        if not os.path.isdir(os.path.expanduser(data_dir)) or force_download:
            download(force=force_download)

        self.labels = labels
        self.data_dir = data_dir
        self.transform = transform

        quake_dirs = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        self.files = []
        for qd in list(filter(lambda q: os.path.isdir(q), quake_dirs)):
            class_dirs = [os.path.join(qd, x) for x in os.listdir(qd) if x in labels]
            for cd in class_dirs:
                sacs = [os.path.join(cd, f) for f in os.listdir(cd) if ('.SAC' in f or '.sac' in f)]
                self.files += sacs

        random.shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]
        data = obspy.read(file)[0]
        label = file.split('/')[-2]

        # one hot encode labels
        one_hot = np.zeros(len(self.labels))
        lbl_idx = self.labels.index(label)
        one_hot[lbl_idx] = 1

        data = {'t': data}

        self.transform(data)

        return data['t'], one_hot