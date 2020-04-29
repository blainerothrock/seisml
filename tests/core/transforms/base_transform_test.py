import pytest
from seisml.core.transforms import BaseTraceTransform, TransformException
import obspy
import numpy as np


class TestTraceBaseTransform:

    def test_initialization(self):
        src = 'test_src'
        out = 'test_out'
        inplace = True
        tf = BaseTraceTransform(source=src, output=out, inplace=inplace)
        assert tf.source == src
        assert tf.output == out
        assert tf.inplace == inplace

    def test_call(self):
        data = {'raw': obspy.Trace(np.array([0, 0, 0]))}
        tf = BaseTraceTransform()
        _ = tf(data)

    def test_inplace(self):
        src = 'test_src'
        out = 'test_out'
        inplace = True

        data = {src: obspy.Trace(np.array([0, 0, 0]))}
        transformed = obspy.Trace(np.array([1, 1, 1]))

        tf = BaseTraceTransform(source=src, output=out, inplace=inplace)
        data = tf.update(data, transformed)
        assert len(data.keys()) == 1
        assert data[src] == transformed

    def test_params(self):
        with pytest.raises(TransformException) as _:
            tf = BaseTraceTransform()
            tf(6)

        with pytest.raises(TransformException) as _:
            tf = BaseTraceTransform(source='test')
            data = {'not_test': []}
            tf(data)

        with pytest.raises(TransformException) as _:
            tf = BaseTraceTransform(source='test')
            data = {'test': [0, 1, 2]}
            tf(data)