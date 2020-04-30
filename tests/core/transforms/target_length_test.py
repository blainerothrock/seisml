import pytest
from seisml.core.transforms import TargetLength, TransformException
import obspy
import numpy as np


class TestTargetLength:

    @pytest.fixture
    def signal(self):
        np.random.seed(1234)
        t = np.linspace(0, 20, 1000)
        x = t + np.random.normal(size=1000)

        return obspy.Trace(x)

    def test_trim(self, signal):
        target_len = 500
        tf = TargetLength(target_length=target_len)
        data = {tf.source: signal}

        tf(data)

        out = data[tf.output].data

        assert len(out) == target_len, 'data should equal target length'

    def test_pad_beginning(self, signal):
        target_len = 1500
        tf = TargetLength(target_length=target_len, random_offset=False)
        data = {tf.source: signal}
        tf(data)
        out = data[tf.output].data

        assert len(out) == target_len, 'data should equal target length'

        diff_half = int((target_len - len(signal.data)) / 2)
        assert np.all(out[:diff_half] == np.zeros(diff_half)), 'trim should be half at beginning'
        assert np.all(out[-diff_half:] == np.zeros(diff_half)), 'trim should be half at end'

    def test_pad_random(self, signal):
        target_len = 750
        diff = target_len - len(signal.data)
        tf = TargetLength(target_length=target_len, random_offset=True)
        data = {tf.source: signal}
        tf(data)
        out = data[tf.output].data

        assert len(out) == target_len, 'data should equal target length'

    def test_params(self):
        with pytest.raises(TransformException):
            tf = TargetLength(target_length=0.1)

    def test_print(self):
        string = str(TargetLength())
        assert 'target_length' in string
