import torch
import os
from ignite.engine import Events, Engine
import numpy as np
from torch.utils.data import DataLoader
from seisml.datasets import TriggeredEarthquake, DatasetMode, triggered_earthquake_transform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


def create_engine(model, optimizer, loss, device):
    model.to(device)

    def _update(engine, batch):
        data, label = batch
        num = data.shape[1]
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
            data = data.view(-1, 1, data.shape[-1])
            data = data.to(device)
            label = label.to(device)
            label = label.float()
            pred = model(data)
            return pred, label

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine



def test_knn(model, testing_quakes, device, data_dir):
    ds_train = TriggeredEarthquake(
        data_dir=data_dir,
        testing_quakes=testing_quakes,
        mode=DatasetMode.TRAIN,
        transform=triggered_earthquake_transform(random_trim_offset=False)
    )
    ds_test = TriggeredEarthquake(
        data_dir=data_dir,
        testing_quakes=testing_quakes,
        mode=DatasetMode.TEST,
        transform=triggered_earthquake_transform(random_trim_offset=False)
    )
    train_loader = DataLoader(ds_train, batch_size=1, num_workers=10)
    test_loader = DataLoader(ds_test, batch_size=1, num_workers=10)

    embeddings = []
    labels = []
    for item in train_loader:
        data, label = item
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        output = model(data).squeeze(1)
        embeddings.append(output.cpu().data.numpy())
        labels.append(label.cpu().data.numpy())

    embeddings = np.vstack(embeddings)
    labels = np.vstack(labels)

    svc = KNeighborsClassifier(n_neighbors=11)
    svc.fit(embeddings, np.argmax(labels, axis=-1))

    preds = []
    ground_truths = []
    for item in test_loader:
        data, label = item
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        output = model(data).squeeze(1)

        embedding = output.cpu().data.numpy()
        label = label.cpu().data.numpy()

        preds.append(svc.predict(embedding))
        ground_truths.append(np.argmax(label, axis=-1))

    cm = confusion_matrix(ground_truths, preds)
    acc = accuracy_score(ground_truths, preds)
    return acc, cm
