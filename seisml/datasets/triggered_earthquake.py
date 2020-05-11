import os, random, shutil
import torch
from torch.utils.data import Dataset
import numpy as np
import obspy
from enum import Enum
from multiprocessing import cpu_count
from seisml.utility.download_data import download_triggered_earthquake_data
from seisml.utility.utils import parallel_process, save_file
from seisml.core.transforms import Resample, \
    ButterworthPassFilter, FilterType, Compose, \
    ToTensor, Augment, AugmentationType, TargetLength


class DatasetMode(str, Enum):
    TRAIN = 'train'
    TEST = 'test'


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
            mode=DatasetMode.TRAIN,
            testing_quakes=[],
            transform=triggered_earthquake_transform()):

        if not os.path.isdir(os.path.expanduser(data_dir)) or force_download:
            download(force=force_download)

        self.labels = labels
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.testing_quakes = testing_quakes

        raw_path = os.path.join(data_dir, 'raw')
        if mode == DatasetMode.TRAIN:
            # include all quakes minus testing quakes
            dirs = filter(lambda d: d not in testing_quakes, os.listdir(raw_path))
            quake_dirs = [os.path.join(raw_path, x) for x in dirs]
        elif mode == DatasetMode.TEST:
            # only include testing quakes
            dirs = filter(lambda d: d in testing_quakes, os.listdir(raw_path))
            quake_dirs = [os.path.join(raw_path, x) for x in dirs]

        self.raw_files = []
        for qd in list(filter(lambda q: os.path.isdir(q), quake_dirs)):
            class_dirs = [os.path.join(qd, x) for x in os.listdir(qd) if x in labels]
            for cd in class_dirs:
                sacs = [os.path.join(cd, f) for f in os.listdir(cd) if ('.SAC' in f or '.sac' in f)]
                self.raw_files += sacs

        self.processed_files = []
        self.preprocess_all_and_save()

    def __len__(self):
        return len(self.raw_files)

    def __getitem__(self, i):
        file = self.processed_files[i]
        p = torch.load(open(file, 'rb'))
        return p['data'], p['label']

    def preprocess_all_and_save(self):
        prepare_path = os.path.join(self.data_dir,
                                    'prepare_train' if self.mode == DatasetMode.TRAIN else 'prepare_test')
        if os.path.isdir(prepare_path) and len(os.listdir(prepare_path)) == len(self.raw_files):
            # files are already preprocessed
            self.processed_files = [os.path.join(prepare_path, f) for f in os.listdir(prepare_path)]
            return

        shutil.rmtree(prepare_path, ignore_errors=True)
        os.mkdir(prepare_path)

        for file in self.raw_files:
            file_split = file.split('/')
            file_name = file_split[-1]
            label = file_split[-2]
            quake = file_split[-3]

            data = obspy.read(file)[0]

            one_hot = np.zeros(len(self.labels), dtype=np.float)
            lbl_idx = self.labels.index(label)
            one_hot[lbl_idx] = 1

            data = {'t': data}

            self.transform(data)

            f = '%s_%s_%s.pt' % (quake, label, file_name)
            torch.save(
                {'data': data['t'], 'label': one_hot},
                open(os.path.join(prepare_path, f), 'wb')
            )
            self.processed_files.append(os.path.join(prepare_path, f))
            random.shuffle(self.processed_files)
