import pytest
from seisml.metrics.loss import DeepClusteringLoss
import torch

def test_deep_clutering_loss():

    # simulating seismic data with 3 sources separated into 2 classes

    batch_size = 24
    num_channel= 2
    num_source = 3
    embedding_size = 10
    num_class = 2

    embedding = torch.rand(batch_size, num_channel, num_source, embedding_size)
    embedding = torch.nn.functional.normalize(embedding, dim=-1, p=2)

    labels = torch.rand(batch_size, num_channel, num_source, num_class) > 0.5
    labels = labels.float()

    weights = torch.ones(batch_size, num_channel, num_source)

    l = DeepClusteringLoss()

    loss_identity = l(embedding, embedding, weights).item()
    assert loss_identity == 0, 'identity loss should equal 0'

    _loss = l(embedding, labels, weights).item()
    assert _loss > loss_identity, 'loss should be greater than zero (identity loss)'
    assert _loss <= 1, 'loss should be less than or equal to 1'