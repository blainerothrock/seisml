import os, pickle, gin, torch, json
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from experiments.utils import create_engine, create_eval, test_classification, create_classifier, get_embeddings
from torch.utils.data import DataLoader, SequentialSampler
from seisml.datasets import TriggeredEarthquake, SiameseDataset, DatasetMode, triggered_earthquake_transform
from seisml.networks import DilatedConvolutional
from seisml.utility.download_data import DownloadableData
from seisml.metrics.loss import DeepClusteringLoss
from torchsummary import summary
import numpy as np
from datetime import datetime


@gin.configurable
def train(
        prefix,
        epochs,
        batch_size,
        num_workers,
        embedding_size,
        num_layers,
        learning_rate,
        weight_decay,
        model_dir,
        run_dir):
    ts = datetime.now().strftime("%m_%d_%Y__%H_%M")
    run_name = '{}_{}'.format(prefix, ts)

    model_dir = os.path.join(model_dir, run_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    writer = SummaryWriter(os.path.join(run_dir, run_name))

    ds_train = TriggeredEarthquake(
        mode=DatasetMode.TRAIN,
        downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE
    )

    ds_test = TriggeredEarthquake(
        mode=DatasetMode.TEST,
        downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE,
        transform=triggered_earthquake_transform(random_trim_offset=False)
    )
    # ds_train = SiameseDataset(ds_train)
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    model = DilatedConvolutional(embedding_size=embedding_size, num_layers=num_layers)
    params = filter(lambda p: p.requires_grad, model.parameters())

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    loss_fn = DeepClusteringLoss()

    trainer = create_engine(model, optimizer, loss_fn, device)
    evaluator = create_eval(model, {'dcl': Loss(loss_fn)}, device)

    summary(model, (1, gin.query_parameter('triggered_earthquake_transform.target_length')))
    writer.add_graph(model, next(iter(train_loader))[0].unsqueeze(1).to(device))

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
        """
        report training loss
        :param _:
        :return:
        """
        evaluator.run(train_loader)
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
        acc, cm, _, = test_classification(
            model,
            gin.query_parameter('triggered_earthquake_dataset.testing_quakes'),
            device,
            gin.query_parameter('triggered_earthquake_dataset.data_dir')
        )
        writer.add_scalar('Accurarcy/test', acc, trainer.state.epoch)
        print('Testing Accurarcy: {:.2f}'.format(acc))
        print(cm)

    def report_embeddings(_):
        """
        write embeddings to tensorboard
        :param _:
        :return:
        """
        train_loader = DataLoader(ds_train, batch_size=1)
        test_loader = DataLoader(ds_test, batch_size=1)

        text_labels = gin.query_parameter('triggered_earthquake_dataset.labels')
        train_embeddings, train_labels = get_embeddings(model, train_loader, device=device)
        train_labels = [text_labels[np.argmax(l)] for l in train_labels.squeeze(1)]
        writer.add_embedding(
            train_embeddings.squeeze(1),
            metadata=train_labels,
            global_step=trainer.state.epoch,
            tag='train_embeddings'
        )

        test_embeddings, test_labels = get_embeddings(model, test_loader, device=device)
        test_labels = [text_labels[np.argmax(l)] for l in test_labels.squeeze(1)]
        writer.add_embedding(
            test_embeddings.squeeze(1),
            metadata=test_labels,
            global_step=trainer.state.epoch,
            tag='test_embeddings'
        )

    trainer.add_event_handler(Events.EPOCH_COMPLETED(once=1), report_embeddings)
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=5), report_embeddings)

    @trainer.on(Events.COMPLETED)
    def save_classifier(_):
        '''
        create and save two svc classifiers in the model_dir
            - one with only training data
            - ont with all data
        :param _:
        :return:
        '''
        # save classifier only trained on training data
        _, _, classifier = test_classification(
            model,
            gin.query_parameter('triggered_earthquake_dataset.testing_quakes'),
            device,
            gin.query_parameter('triggered_earthquake_dataset.data_dir')
        )
        with open(os.path.join(model_dir, '{}_classifier.p'.format(prefix)), 'wb') as f:
            pickle.dump(classifier, f)

        # save classifier trained on all data (for running inference)
        ds = TriggeredEarthquake(
            data_dir=gin.query_parameter('triggered_earthquake_dataset.data_dir'),
            testing_quakes=[],
            downloadable_data=DownloadableData.TRIGGERED_EARTHQUAKE,
            mode=DatasetMode.INFERENCE,
            transform=triggered_earthquake_transform(random_trim_offset=False),
        )
        loader = DataLoader(ds, batch_size=1, num_workers=10, shuffle=True)
        classifier_alldata = create_classifier(model, loader, type='svc', device=device)
        with open(os.path.join(model_dir, '{}_svc_classifier.p'.format(prefix)), 'wb') as f:
            pickle.dump(classifier_alldata, f)

    @trainer.on(Events.COMPLETED)
    def save_metadata(_):
        '''
        save a metadata file, used for inference
        :param _:
        :return:
        '''
        transformer = triggered_earthquake_transform(random_trim_offset=False)
        transformer_path = os.path.join(model_dir, 'transformer.p')
        pickle.dump(transformer, open(transformer_path, 'wb'))

        metadata = {
            'name': run_name,
            'classes': gin.query_parameter('triggered_earthquake_dataset.labels'),
            'model_state_path': save_handler.last_checkpoint,
            'classifier_path': os.path.join(model_dir, '{}_classifier.p'.format(prefix)),
            'embedding_size': embedding_size,
            'num_layers': num_layers,
            'transformer': transformer_path
        }

        with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    trainer.run(train_loader, max_epochs=epochs)
    writer.close()


if __name__ == '__main__':
    gin.parse_config_file('config.gin')
    os.chdir('../../')
    print('current working directory: %s' % os.getcwd())
    train()
