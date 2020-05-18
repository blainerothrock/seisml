import pytest, os
from seisml.datasets import TriggeredEarthquake, DatasetMode, SiameseDataset, triggered_earthquake_transform
from seisml.utility.download_data import DownloadableData
from seisml.networks import DilatedConvolutional
from seisml.metrics.loss import DeepClusteringLoss
from torch.utils.data import DataLoader
import torch


@pytest.mark.non_ci
class TestTriggeredEarthquakeTraining:

    @pytest.fixture()
    def train_dataset(self):
        transform = triggered_earthquake_transform(
            sampling_rate=20.0,
            max_freq=8.0,
            min_freq=2.0,
            corner=2,
            aug_types=None,
            aug_prob=0.5,
            target_length=8192,
            random_trim_offset=True
        )

        ds = TriggeredEarthquake(
            data_dir=os.path.expanduser('~/.seisml/data/triggered_earthquakes'),
            force_download=False,
            downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE,
            labels=['positive', 'negative'],
            mode=DatasetMode.TRAIN,
            testing_quakes=['SAC_20021102_XF_prem'],
            transform=transform)
        return ds

    def test_batch_distrubution(self, train_dataset):
        batch_size = 32

        dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

        for data, label in dl:
            negative_count = len(list(filter(lambda x: x[1] == 1, label)))
            positive_count = len(list(filter(lambda x: x[0] == 1, label)))
            pos_percent = int((positive_count/batch_size) * 100)
            neg_percent = int((negative_count/batch_size) * 100)
            print('pos:neg: {}:{}'.format(pos_percent, neg_percent))
            assert pos_percent > 20, 'should have at least 1/5 class representation'
            assert neg_percent > 20, 'should have at least 1/5 class representation'


        # assert negative_count + positive_count == batch_size
        # assert negative_count == batch_size / 2, 'should be even split'
        # assert positive_count == batch_size / 2, 'should be even split'

    def test_label_matching(self, train_dataset):
        files = train_dataset.processed_files

        assert len(files) == len(set(files)), 'all observations should be unique'

        for f in files:
            data = torch.load(open(f, 'rb'))
            label = data['label']
            assert sum(label) == 1, 'should be one label'
            quake = data['quake']
            print(f.split('_')[-1].split('.pt')[0], label, quake)


    def test_overfit(self, train_dataset):
        batch_size = 128
        epochs = 100
        dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=1)
        batch = next(iter(dl))
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model = DilatedConvolutional(embedding_size=10)
        params = filter(lambda p: p.requires_grad, model.parameters())
        model.to(device)

        optimizer = torch.optim.Adam(params, lr=2e-5, weight_decay=0.1)
        loss = DeepClusteringLoss()

        loss_history = []

        for epoch in range(epochs):
            data, label = batch
            for i in range(data.shape[0]):

                obs = data[i].unsqueeze(0)
                tar = label[i].unsqueeze(0)

                num = 1
                data = data.view(-1, 1, data.shape[-1])
                data = data.to(device)
                label = label.to(device)
                label = label.float()

                model.train()

                optimizer.zero_grad()
                output = model(data)
                output = output.view(-1, num, output.shape[-1])
                _loss = loss(output, label)
                _loss.backward()
                optimizer.step()
                loss_history.append(_loss.item())

            print('- {}: loss: {:.2f}'.format(epoch, _loss.item()))

        print(loss_history)



