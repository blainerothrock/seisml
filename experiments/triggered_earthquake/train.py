import os
from ignite.engine import Events, Engine
from ignite.metrics import Loss
import torch
from torch.utils.tensorboard import SummaryWriter
from triggered_earthquake_engine import create_engine, create_eval, test_knn
from torch.utils.data import DataLoader, SequentialSampler
from seisml.datasets import TriggeredEarthquake, SiameseDataset, DatasetMode, triggered_earthquake_transform
from seisml.networks import DilatedConvolutional
from seisml.utility.download_data import DownloadableData
from seisml.metrics.loss import DeepClusteringLoss
from torchsummary import summary
import gin


@gin.configurable
def train(
        epochs,
        batch_size,
        num_workers,
        learning_rate,
        weight_decay):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    writer = SummaryWriter()

    ds_train = TriggeredEarthquake(
        mode=DatasetMode.TRAIN,
        downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE
    )

    ds_test = TriggeredEarthquake(
        mode=DatasetMode.TEST,
        downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE,
        transform=triggered_earthquake_transform(random_trim_offset=False)
    )
    ds_train = SiameseDataset(ds_train)
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, num_workers=num_workers, sampler=SequentialSampler(ds_train))
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, num_workers=num_workers, sampler=SequentialSampler(ds_test))

    model = DilatedConvolutional(embedding_size=10)
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    loss = DeepClusteringLoss()

    trainer = create_engine(model, optimizer, loss, device)
    evaluator = create_eval(model, {'loss': Loss(loss)}, device)

    summary(model, (1,8192))

    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(trainer):
    #     print("Epoch[{}] Loss: {:.2f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        writer.add_scalar('Loss/train', metrics['loss'], trainer.state.epoch)
        print("Training Results - Epoch: {} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        writer.add_scalar('Loss/test', metrics['loss'], trainer.state.epoch)
        print("Testing Results - Epoch: {} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def test_acc(trainer):
        acc, cm = test_knn(
            model,
            gin.query_parameter('triggered_earthquake_dataset.testing_quakes'),
            device,
            gin.query_parameter('triggered_earthquake_dataset.data_dir')
        )
        writer.add_scalar('Accurarcy/test', acc, trainer.state.epoch)
        print('Testing Accurarcy: {:.2f}'.format(acc))
        print(cm)

    trainer.run(train_loader, max_epochs=epochs)


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    os.chdir('../../')
    print('current working directory: %s' % os.getcwd())
    train()
