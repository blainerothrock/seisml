import os
from torch.utils.data import Dataset
import numpy as np
import obspy
from multiprocessing import cpu_count
from seisml.utility.download_data import download_triggered_earthquake_data
from seisml.utility.utils import parallel_process, save_file


class TriggeredEarthquake(Dataset):
    """
    Dataset for Triggered Earthquakes as refferenced in Automating the Detection of Dynamically Triggered Earthquakes
    via a Deep Metric Learning Algorithm

    Args:
        folder: (str): folder where raw triggered earthquake data exists
        force_download: (Bool): [optional] if data already exists, re-download it, default: False
        download: (function): [optional] download method, default: download_triggered_earthquake_data
    """

    def __init__(
            self,
            folder='~/.seisml/data/triggered_earthquakes',
            force_download=False,
            download=download_triggered_earthquake_data):

        if not os.path.isdir(os.path.expanduser(folder)) or force_download:
            download(force=force_download)

        if not os.path.isdir(os.path.expanduser(folder + '/prepared')):
            self.preprocess(os.path.expanduser(folder))

    def process_file(self, earthquake_file, directory, target_directory, label):
        earthquake = obspy.read(os.path.join(directory, earthquake_file))[0]
        earthquake.resample(sampling_rate=20, window='hanning', no_filter=True, strict_length=False)
        data_dict = {}

        data_dict['data'] = earthquake
        data_dict['label'] = label
        data_dict['name'] = directory.split('/')[-2]

        save_file(
            os.path.join(
                target_directory,
                '%s_%s_%s.p' % (label, data_dict['name'],
                                earthquake_file)
            ),
            data_dict
        )

    def load_data(self, directory, label, target_directory):
        if not os.path.exists(directory):
            return []

        earthquake_files = []
        for file in os.listdir(directory):
            if not ('.SAC' in file or '.sac' in file): continue
            print(file)
            earthquake_files.append({
                'earthquake_file': file,
                'directory': directory,
                'target_directory': target_directory,
                'label': label
            })

        print(earthquake_files)

        if len(earthquake_files) == 0:
            return []

        parallel_process(earthquake_files, self.process_file, use_kwargs=True, n_jobs=min(cpu_count() - 1, 1))

    def preprocess(self, folder, raw_dir='raw'):
        quake_folders = [os.path.join(folder + '/' + raw_dir, x) for x in os.listdir(folder + '/' + raw_dir)]
        accepted_labels = ['positive', 'negative']
        output_directory = folder + '/prepared'
        os.makedirs(output_directory, exist_ok=True)

        for qf in quake_folders:
            if not os.path.isdir(qf): continue
            labels = labels = [x for x in os.listdir(qf) if x in accepted_labels]
            for l in labels:
                self.load_data(os.path.join(qf, l), l, output_directory)
