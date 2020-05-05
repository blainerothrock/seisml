import pytest
import numpy as np

from seisml.core.transforms import CustomTraceTransform

class TestCustomTraceTransform:

    def test_transform(self, signal):
        data = {'raw': signal}

        def transform(trace):
            return trace.data * 2

        id = 'mul'
        tf = CustomTraceTransform(transform, identifier=id)
        tf(data)
        out = data[tf.output].data

        assert np.allclose(signal.data * 2, out), 'output should match'

    def test_print(self):
        id = 'test_'
        string = str(CustomTraceTransform(lambda t: t.data * 2, identifier=id))
        assert id in string
