import os, pickle, gin, torch, json
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from experiments.utils import create_engine, create_eval, test_classification, create_classifier, get_embeddings, report_accurarcy
from torch.utils.data import DataLoader, SequentialSampler
from seisml.datasets import triggered_tremor_split
from seisml.networks import DilatedConvolutional
from seisml.utility.download_data import DownloadableData
from seisml.metrics.loss import DeepClusteringLoss, WhitenedKMeansLoss
from torchsummary import summary
import numpy as np
from datetime import datetime

@gin.configurable()
def train(
        prefix,
        epochs,
        embedding_size,
        num_layers,
        learning_rate,
        weight_decay,
        model_dir,
        run_dir):

    device = torch.device('cuda' if torch.cuda else 'cpu')

    ts = datetime.now().strftime("%m_%d_%Y__%H_%M")
    run_name = '{}_{}'.format(prefix, ts)

    writer = SummaryWriter(os.path.join(run_dir, run_name))

    train_dl, test_dl = triggered_tremor_split()

    model = DilatedConvolutional(embedding_size=embedding_size, num_layers=num_layers, downsample=False)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = DeepClusteringLoss()

    trainer = create_engine(model, optimizer, loss_fn, device)
    evaluator = create_eval(model, {'dcl': Loss(loss_fn)}, device)

    summary(model, (1, 100000))
    writer.add_graph(model, next(iter(train_dl))[0].unsqueeze(1).to(device))

    save_handler = ModelCheckpoint(model_dir, prefix, n_saved=1, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'model': model})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(_):
        """
        report training loss
        :param _:
        :return:
        """
        writer.add_scalar('Iter/train_loss', trainer.state.output, trainer.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(_):
        evaluator.run(train_dl)
        loss = trainer.state.output
        writer.add_scalar('Loss/train', loss, trainer.state.epoch)
        print("Training Results - Epoch: {} Avg loss: {:.2f}"
              .format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def test_acc(_):
        """
        report testing accurarcy
        :param _:
        :return:
        """
        svc = create_classifier(model, train_dl, type='svc', device=device)
        acc, cm = report_accurarcy(model, svc, test_dl, device=device)

        writer.add_scalar('Accurarcy/test', acc, trainer.state.epoch)
        print('Testing Accurarcy: {:.2f}'.format(acc))
        print(cm)


    trainer.run(train_dl, max_epochs=epochs)
    writer.close()


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    os.chdir('../../')
    print('current working directory: %s' % os.getcwd())
    train()