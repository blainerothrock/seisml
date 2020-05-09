from torch.utils.data import Dataset
import numpy as np


class SiameseDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        data_one, label_one = self.dataset[i]
        data_two, label_two = self.dataset[np.random.randint(0, len(self.dataset))]

        while np.argmax(label_two) == np.argmax(label_one):
            data_two, label_two = self.dataset[np.random.randint(0, len(self.dataset))]

        data = np.vstack([data_one, data_two])
        label = np.vstack([label_one, label_two])
        # weights = np.vstack([item_one['weight'], item_two['weight']])
        return data, label

    def __len__(self):
        return len(self.dataset)
