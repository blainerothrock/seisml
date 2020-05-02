import pytest
from seisml.core.transforms import ToTensor, TransformException
import obspy
import torch
import numpy as np


class TestToTensor:

    def test_to_tensor(self, signal):
        device = torch.device('cpu')
        tf = ToTensor(device=device)
        data = {tf.source: signal}
        tf(data)
        out = data[tf.output]

        assert isinstance(out, torch.Tensor), 'should be tensor'
        assert np.allclose(signal.data, out.numpy()), 'tensor should be equal to original signal'

    def test_params(self):
        with pytest.raises(TransformException):
            tf = ToTensor(device=0)

    def test_print(self):
        string = str(ToTensor(device=torch.device('cpu')))
        assert 'device' in string