import pytest
from seisml.core.transforms import ToTensor, TransformException
import obspy
import torch
import numpy as np


class TestToTensor:

    def test_to_tensor(self, signal):
        device = torch.device('cpu')
        tf = ToTensor()
        data = {tf.source: signal}
        tf(data)
        out = data[tf.output]

        assert isinstance(out, torch.Tensor), 'should be tensor'
        assert np.allclose(signal.data, out.numpy()), 'tensor should be equal to original signal'

    def test_print(self):
        string = str(ToTensor())
        assert 'ToTensor' in string