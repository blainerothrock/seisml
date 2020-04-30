import pytest
from seisml.core.transforms import Normalize, TransformException
import obspy
import numpy as np


class TestNormalize:

    @pytest.fixture
    def signal(self):
        np.random.seed(1234)
        t = np.linspace(0, 20, 100)
        x = t + np.random.normal(size=100)

        return obspy.Trace(x)

    def test_normalization(self, signal):
        tf = Normalize()
        data = {tf.source: signal}

        tf(data)

        out = data[tf.output].data

        assert np.isclose(out.mean(), 0), 'mean should be zero'
        assert np.isclose(out.std(), 1), 'standard deviation should be 1'

    def test_print(self):
        string = str(Normalize())
        assert string is not None
        assert string == 'Normalize()'
