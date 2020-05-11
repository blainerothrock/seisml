import os
from ignite.engine import Events, Engine
from ignite.metrics import Loss
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.utils.data import DataLoader
from seisml.datasets import TriggeredEarthquake, SiameseDataset
from seisml.networks import DilatedConvolutional
from seisml.metrics.loss import DeepClusteringLoss

os.chdir('../../')
print('current working directory: %s' % os.getcwd())


def create_clustering_engine(model, optimizer, loss, device):
    model.to(device)

    def _update(engine, batch):
        data, label = batch
        data = data.view(-1, 1, data.shape[-1])
        data = data.to(device)
        label = label.to(device)
        label = label.float()
        model.train()

        optimizer.zero_grad()
        output = model(data)
        _loss = loss(output, label)
        _loss.backward()
        optimizer.step()

        return _loss.item()

    return Engine(_update)


def create_clustering_eval(model, metrics, device):
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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ds = TriggeredEarthquake()
# ds = SiameseDataset(ds)
train_loader = DataLoader(ds, batch_size=64, num_workers=10)

model = DilatedConvolutional(embedding_size=10)
params = filter(lambda p: p.requires_grad, model.parameters())

optimizer = torch.optim.Adam(params, lr=2e-5, weight_decay=1e-1)
loss = DeepClusteringLoss()

trainer = create_clustering_engine(model, optimizer, loss, device)
evaluator = create_clustering_eval(model, {'loss': Loss(loss)}, device)


# @trainer.on(Events.ITERATION_COMPLETED)
# def log_training_loss(trainer):
#     print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {} Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['loss']))


trainer.run(train_loader, max_epochs=100)
