import pytest
import torch
from seisml.core.transforms import Compose, TransformException, ButterworthPassFilter, ToTensor, Normalize, FilterType

class TestCompose:

    @pytest.fixture
    def compose(self, sample_trace):
        data = {'raw': sample_trace}

        tf = Compose([
            ButterworthPassFilter(filter_type=FilterType.BANDPASS, min_freq=0.1, max_freq=5.0, corners=2),
            Normalize(),
            ToTensor()
        ])

        return tf

    def test_compose(self, sample_trace, compose):
        data = {'raw': sample_trace}

        tf = Compose([
            ButterworthPassFilter(filter_type=FilterType.BANDPASS, min_freq=0.1, max_freq=5.0, corners=2),
            Normalize(),
            ToTensor()
        ])

        tf(data)
        assert isinstance(data[tf.output], torch.Tensor), 'should be tensor'

    def test_print(self, compose):
        string = str(compose)
        assert 'Compose' in string
        assert 'ButterworthPassFilter' in string
        assert 'Normalize' in string
        assert 'ToTensor' in string