import pytest
from seisml.core.transforms import Normalize, TransformException
import obspy
import numpy as np


class TestNormalize:

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
