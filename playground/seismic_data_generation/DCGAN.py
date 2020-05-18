import os
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data


class DCAGN:

    def __init__(self):
        print()

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label, length, mode):
        self.data_dir = data_dir
        self.files = list(filter(lambda f: f.startswith(label), os.listdir(data_dir)))
        self.length = length
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        with open(path, 'rb') as f:
            earthquake = pickle.load(f)

        sac = earthquake['data']
        data = np.stack(sac, axis=0)

        data = self.get_target_length_and_transpose(data, self.length)
        weight = 1.  # / self.priors[index]

        # data = self.post_transform(data, self.transforms)

        return {'data': data, 'weight': np.sqrt(weight)}

    def get_target_length_and_transpose(self, data, target_length):
        length = data.shape[-1]
        if target_length == 'full':
            target_length = length
        if length > target_length:
            if self.mode == 'train':
                offset = np.random.randint(0, length - target_length)
            else:
                offset = 0
        else:
            offset = 0
        pad_length = max(target_length - length, 0)
        pad_tuple = [(0, 0) for k in range(len(data.shape))]
        pad_tuple[0] = (int(pad_length / 2), int(pad_length / 2) + (length % 2))
        data = np.pad(data, pad_tuple, mode='constant')
        data = data[offset:offset + target_length]
        return data
