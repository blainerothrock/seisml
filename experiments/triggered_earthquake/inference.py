import torch, gin, os, pickle, json, csv
import numpy as np
from datetime import datetime
from seisml.datasets import TriggeredEarthquake, DatasetMode, triggered_earthquake_transform
from seisml.networks import DilatedConvolutional
from utils import get_embeddings
from torch.utils.data import DataLoader


def inference(experiment_path, earthquake_path, labels=None):

    ts = datetime.now().strftime("%m_%d_%Y__%H_%M")
    result_name = '{}_{}'.format(earthquake_path.split('/')[-1], ts)

    # load metadata
    with open(os.path.join(experiment_path, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    model_path = metadata['model_state_path']
    classifier_path = metadata['classifier_path']
    embedding_size = metadata['embedding_size']
    num_layers = metadata['num_layers']
    transformer_path = metadata['transformer']

    # load the model
    state = torch.load(model_path)
    model = DilatedConvolutional(embedding_size=embedding_size, num_layers=num_layers)
    model.load_state_dict(state)

    # load the classifier
    classifier = pickle.load(open(classifier_path, 'rb'))

    # run through each example in the earthquake path
    transformer = pickle.load(open(transformer_path, 'rb'))
    dataset = TriggeredEarthquake(
        data_dir=earthquake_path,
        downloadable_data=None,
        mode=DatasetMode.INFERENCE,
        testing_quakes=[],
        labels=labels,
        transform=transformer
    )

    result_csv_path = os.path.join(experiment_path, '{}_results.csv'.format(result_name))
    headers = ['quake', 'name', 'given_label', 'classification']
    writer = csv.DictWriter(open(result_csv_path, 'w'), fieldnames=headers)
    writer.writeheader()

    device = torch.device('cuda' if torch.cuda else 'cpu')
    model.to(device)

    embeddings = []
    for obs in dataset.processed_files:
        processed = torch.load(obs)
        data = processed['data']
        label = processed['label']
        quake = processed['quake']
        file_name = processed['file_name']

        embedding = model(data.view(-1, 1, data.shape[-1]).to(device)).detach().cpu().numpy()
        embeddings.append(embeddings)

        classification = labels[np.argmax(classifier.predict(embedding))]
        writer.writerow({
            'quake': quake,
            'name': file_name,
            'given_label': label,
            'classification': classification
        })



if __name__ == '__main__':
    inference(
        os.path.expanduser('~/.seisml/experiments/triggered_earthquakes/test_infer_05_26_2020__16_40/'),
        os.path.expanduser('~/.seisml/data/te_test_inference/'),
        labels=['positive', 'negative']
    )
