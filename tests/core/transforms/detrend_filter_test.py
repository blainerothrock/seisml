import pytest
from seisml.core.transforms import DetrendType, DetrendFilter, TransformException
import obspy
import numpy as np

class TestDetrendFilter:

    def test_linear(self, signal_with_linear):

        tf = DetrendFilter(detrend_type=DetrendType.LINEAR)
        data = { 'raw': signal_with_linear }

        tf(data)

        out = data[tf.output].data

        assert out.mean() < signal_with_linear.data.mean(), 'mean should be reduced'
        assert np.isclose(out.mean(), 0), 'mean should be close to 0'

    def test_params(self):
        with pytest.raises(TransformException):
            tf = DetrendFilter(detrend_type=0)

    def test_print(self):
        string = str(DetrendFilter())
        assert 'detrend_type' in string