import os, pickle, gin, torch, json
from ignite.engine import Events, Engine, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import Loss, Accuracy
from ignite.handlers import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from seisml.datasets import triggered_tremor_split
from seisml.networks import ConvNet
from torchsummary import summary
import numpy as np
from datetime import datetime

@gin.configurable()
def train(
        prefix,
        epochs,
        learning_rate,
        model_dir,
        run_dir):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = datetime.now().strftime("%m_%d_%Y__%H_%M")
    run_name = '{}_{}'.format(prefix, ts)

    writer = SummaryWriter(os.path.join(run_dir, run_name))

    train_dl, test_dl = triggered_tremor_split()

    model = ConvNet(input_shape=(1, 20000), num_layers=3, hidden_dims=(8, 16, 32), pool_factor=(2, 2, 2), conv_kernel=2, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss = torch.nn.CrossEntropyLoss()

    def prepare_batch(batch, device, non_blocking):
        data, label = batch
        data = data.unsqueeze(1).to(device)
        label = torch.argmax(label.squeeze(1), dim=1).to(device)

        return data, label

    trainer = create_supervised_trainer(model, optimizer, loss, prepare_batch=prepare_batch, device=device)
    evaluator = create_supervised_evaluator(
        model,
        metrics={'accuracy': Accuracy(), 'loss': Loss(loss)},
        prepare_batch=prepare_batch,
        device=device,
    )

    summary(model, (1, 20000))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainier):
        writer.add_scalar('Iter/train_loss', trainer.state.output, trainer.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_dl)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        accuracy = metrics['accuracy']
        print('Epoch {}:'.format(trainer.state.epoch))
        print('  - train accuracy: {:.2f}'.format(accuracy))
        print('  - train loss: {:.4f}'.format(loss))
        writer.add_scalar('Loss/train', loss, trainer.state.epoch)
        writer.add_scalar('Accuracy/train', accuracy, trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(trainer):
        evaluator.run(test_dl)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        accuracy = metrics['accuracy']
        print('Epoch {}:'.format(trainer.state.epoch))
        print('  - test accuracy: {:.2f}'.format(accuracy))
        print('  - test loss: {:.4f}'.format(loss))
        writer.add_scalar('Loss/test', loss, trainer.state.epoch)
        writer.add_scalar('Accuracy/test', accuracy, trainer.state.epoch)

    trainer.run(train_dl, max_epochs=epochs)


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    os.chdir('../../../')
    print('current working directory: %s' % os.getcwd())
    train()