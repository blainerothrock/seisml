import torch
import os
from ignite.engine import Events, Engine
import numpy as np
from torch.utils.data import DataLoader
from seisml.datasets import TriggeredEarthquake, DatasetMode, triggered_earthquake_transform
from seisml.utility.download_data import DownloadableData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC


def create_engine(model, optimizer, loss, device):
    """
    creates pytorch ignite Engine
    :param model: pytorch model
    :param optimizer: pytorch optimizer
    :param loss: a loss function
    :param device: pytorch device
    :return: ignite.engine.Engine
    """
    model.to(device)

    def _update(engine, batch):
        data, label = batch
        num_channels = 1 if len(data.shape) == 2 else data.shape[1]
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        label = label.float()

        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        output = model(data)
        output = output.view(-1, num_channels, output.shape[-1])
        _loss = loss(output, label)
        _loss.backward()
        optimizer.step()

        return _loss.item()

    return Engine(_update)


def create_eval(model, metrics, device):
    """
    Create pytorch ignite evaluator
    :param model: pytorch model
    :param metrics: metrics dictionary
    :param device: pytroch device
    :return: ignite.engine.Engine
    """
    metrics = metrics or {}

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, label = batch
            num_channels = 1 if len(data.shape) == 2 else data.shape[1]
            data = data.view(-1, 1, data.shape[-1])
            data = data.to(device)
            label = label.to(device)
            label = label.float()

            output = model(data)
            output = output.view(-1, num_channels, output.shape[-1])

            return output, label

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def get_embeddings(model, loader, device=torch.device('cpu')):
    """
    Creates embeddings for every item in the loader
    :param model: trained pytorch model
    :param loader: torch.utils.data.Dataloader
    :param device: pytorch device
    :return: embeddings (np.array), labels (np.array)
    """
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
        embeddings.append(embedding)
        labels.append(label)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def test_classification(model, testing_quakes, device, data_dir):
    """
    test model on testing set
    :param model: trained pytorch model
    :param testing_quakes: Array of quakes to test
    :param device: pytorch device
    :param data_dir: direcory of triggered earthquake data
    :return: accurarcy, confusion matrix string, classification model
    """
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
    """
    Create a classifier using scikit learn over a training set.
    :param model: trained pytorch model
    :param loader: torch.utils.data.Dataloader
    :param type: string, 'svc' or 'knn'
    :param device: pytorch device
    :return: classification model
    """
    embeddings, labels = get_embeddings(model, loader, device)
    embeddings = embeddings.squeeze(1)
    labels = labels.squeeze(1)

    if type == 'knn':
        svc = KNeighborsClassifier(n_neighbors=11)
    else:
        svc = SVC()

    svc.fit(embeddings, np.argmax(labels, axis=-1))

    return svc


def report_accurarcy(model, svc, loader, device=torch.device('cpu')):
    """
    Report accurarcy over a data loader
    :param model: a trained pytorch model
    :param svc: classification model (scikit-learn)
    :param loader: torch.utils.data.Dataloader
    :param device: pytorch device
    :return: accurarcy (float), confusion matrix (string)
    """
    embeddings, labels = get_embeddings(model, loader, device)

    predictions = [svc.predict(e) for e in embeddings]
    ground_truths = [np.argmax(l, axis=-1) for l in labels]

    cm = confusion_matrix(ground_truths, predictions)
    acc = accuracy_score(ground_truths, predictions)
    return acc, cm
