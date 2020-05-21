import torch
import os
from ignite.engine import Events, Engine
import numpy as np
from torch.utils.data import DataLoader
from seisml.datasets import TriggeredEarthquake, DatasetMode, triggered_earthquake_transform
from seisml.utility.download_data import DownloadableData
from seisml.utility.utils import visualize_embedding
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA


def create_engine(model, optimizer, loss, device):
    model.to(device)

    def _update(engine, batch):
        data, label = batch
        # num = data.shape[1]
        num = 1
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        label = label.float()

        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        output = model(data)
        output = output.view(-1, num, output.shape[-1])
        _loss = loss(output, label)
        _loss.backward()
        optimizer.step()

        return _loss.item()

    return Engine(_update)


def create_eval(model, metrics, device):
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, label = batch
            # num = data.shape[1]
            num = 1
            data = data.view(-1, 1, data.shape[-1])
            data = data.to(device)
            label = label.to(device)
            label = label.float()

            output = model(data)
            output = output.view(-1, num, output.shape[-1])

            return output, label

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def test_classification(model, testing_quakes, device, data_dir):
    ds_train = TriggeredEarthquake(
        data_dir=data_dir,
        testing_quakes=testing_quakes,
        downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE,
        mode=DatasetMode.INFERENCE,
        transform=triggered_earthquake_transform(random_trim_offset=False),
    )
    ds_test = TriggeredEarthquake(
        data_dir=data_dir,
        testing_quakes=testing_quakes,
        downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE,
        mode=DatasetMode.TEST,
        transform=triggered_earthquake_transform(random_trim_offset=False)
    )
    train_loader = DataLoader(ds_train, batch_size=1, num_workers=10, shuffle=True)
    test_loader = DataLoader(ds_test, batch_size=1, num_workers=10, shuffle=True)

    svc = create_classifier(model, train_loader, type='svc', device=device)
    acc, cm = report_accurarcy(model, svc, test_loader, device=device)

    return acc, cm, svc


def create_classifier(model, loader, type='svc', device=torch.device('cpu')):
    embeddings = []
    labels = []
    for item in loader:
        data, label = item
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        output = model(data).squeeze(1)
        embeddings.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels)

    if type == 'knn':
        svc = KNeighborsClassifier(n_neighbors=11)
    else:
        svc = SVC()

    svc.fit(embeddings, np.argmax(labels, axis=-1))

    return svc


def report_accurarcy(model, svc, loader, device=torch.device('cpu')):
    embeddings = []
    preds = []
    ground_truths = []
    for item in loader:
        data, label = item
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        output = model(data).squeeze(1)

        embedding = output.cpu().data.numpy()
        label = label.cpu().data.numpy()

        embeddings.append(embeddings)
        preds.append(svc.predict(embedding))
        ground_truths.append(np.argmax(label, axis=-1))

    cm = confusion_matrix(ground_truths, preds)
    acc = accuracy_score(ground_truths, preds)
    return acc, cm


def embeddings(model, loader, device=torch.device('cpu')):
    embeddings = []
    labels = []
    for item in loader:
        data, label = item
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        output = model(data).squeeze(1)

        embedding = output.cpu().data.numpy()
        label = label.cpu().data.numpy()
        embeddings.append(embedding[0])
        labels.append(label[0])

    return embeddings, labels
