from torch.utils.data import Dataset
import numpy as np


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        item_one = self.dataset[i]
        item_two = self.dataset[np.random.randint(0, len(self.dataset))]

        while np.argmax(item_two['label']) == np.argmax(item_one['label']):
            item_two = self.dataset[np.random.randint(0, len(self.dataset))]

        data = np.vstack([item_one['data'], item_two['data']])
        label = np.vstack([item_one['label'], item_two['label']])
        weights = np.vstack([item_one['weight'], item_two['weight']])
        return {'data': data, 'label': label, 'weight': weights}

    def __len__(self):
        return len(self.dataset)
