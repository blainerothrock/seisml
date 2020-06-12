import gin
import os
import torch
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from seisml.datasets import MarsInsight, mars_insight_transform
from seisml.utility.utils import split_dataset
from seisml.networks import ConvAutoEncoder
import numpy as np
from datetime import datetime
from torchsummary import summary

@gin.configurable()
def train(
        prefix,
        epochs,
        learning_rate,
        model_dir,
        run_dir):

    ts = datetime.now().strftime("%m_%d_%Y__%H_%M")
    run_name = '{}_{}'.format(prefix, ts)

    model_dir = os.path.join(model_dir, run_name)

    writer = SummaryWriter(os.path.join(run_dir, run_name))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = MarsInsight()
    dl_train, dl_test = split_dataset(ds)

    model = ConvAutoEncoder().to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(params, lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    summary(model, next(iter(dl_train))[1:])

    def update_model(trainer, X):
        X = X.to(device)

        optimizer.zero_grad()

        output, embedding = model(X)
        loss = loss_fn(output, X)
        loss.backward()
        optimizer.step()
        return loss.item(), embedding

    trainer = Engine(update_model)

    save_handler = ModelCheckpoint(model_dir, prefix, n_saved=1, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, save_handler, {'model': model})

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iteration_loss(_):
        loss, _ = trainer.state.output
        writer.add_scalar('Iter/train_loss',loss, trainer.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_loss(_):
        loss, _ = trainer.state.output
        writer.add_scalar('Loss/train', loss, trainer.state.epoch)
        print("Epoch: {} Avg loss: {:.2f}"
              .format(trainer.state.epoch, trainer.state.output))

    trainer.run(dl_train, max_epochs=epochs)

if __name__ == '__main__':
    gin.parse_config_file('gin.config')
    train()

