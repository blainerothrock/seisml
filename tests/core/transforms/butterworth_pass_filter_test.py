import pytest
from seisml.core.transforms import ButterworthPassFilter, FilterType, TransformException
import obspy
import numpy as np


class TestButterworthPassFilter:

    @pytest.fixture
    def signal(self):
        np.random.seed(1234)
        time_step = 0.02
        period = 5.
        time_vec = np.arange(0, 20, time_step)
        sig = (np.sin(2 * np.pi / period * time_vec)
               + 0.5 * np.random.randn(time_vec.size))
        return obspy.Trace(sig)

    def test_lowpass(self, signal):
        filter_type = FilterType.LOWPASS
        tf = ButterworthPassFilter(filter_type=filter_type, max_freq=0.1, corners=1)

        data = {'raw': signal}
        src = data[tf.source].data

        tf(data)

        out = data[tf.output].data
        assert not np.allclose(src, out), 'output should be transformed'
        assert len(src) == len(out), 'input and output length should be equal'

    def test_highpass(self, signal):
        filter_type = FilterType.HIGHPASS
        tf = ButterworthPassFilter(filter_type=filter_type, min_freq=0.1, corners=2)

        data = {'raw': signal}
        src = data[tf.source].data

        tf(data)

        out = data[tf.output].data
        assert not np.allclose(src, out), 'output should be transformed'
        assert len(src) == len(out), 'input and output length should be equal'

    def test_bandpass(self, signal):
        filter_type = FilterType.BANDPASS
        tf = ButterworthPassFilter(filter_type=filter_type, min_freq=0.1, max_freq=0.2, corners=2)

        data = {'raw': signal}
        src = data[tf.source].data

        tf(data)

        out = data[tf.output].data
        assert not np.allclose(src, out), 'output should be transformed'
        assert len(src) == len(out), 'input and output length should be equal'

    def test_bandstop(self, signal):
        filter_type = FilterType.BANDSTOP
        tf = ButterworthPassFilter(filter_type=filter_type, min_freq=0.3, max_freq=0.4, corners=2)

        data = {'raw': signal}
        src = data[tf.source].data

        tf(data)

        out = data[tf.output].data
        assert not np.allclose(src, out), 'output should be transformed'
        assert len(src) == len(out), 'input and output length should be equal'

    def test_params(self):
        with pytest.raises(TransformException):
            tf = ButterworthPassFilter(filter_type=0)

        with pytest.raises(TransformException):
            tf = ButterworthPassFilter(min_freq=int(100))

        with pytest.raises(TransformException):
            tf = ButterworthPassFilter(max_freq=int(100))

        with pytest.raises(TransformException):
            tf = ButterworthPassFilter(corners=0.01)

        with pytest.raises(TransformException):
            tf = ButterworthPassFilter(zerophase=100)

    def test_print(self):
            string = str(ButterworthPassFilter())
            assert 'filter_type' in string

