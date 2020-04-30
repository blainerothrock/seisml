import pytest
from seisml.core.transforms import TrimSurfaceWindow, TransformException
import obspy
import numpy as np

class TestTrimSurfaceWindow:

    def test_trim(self, sample_trace):
        sv = 5.0
        ev = 2.5
        tf = TrimSurfaceWindow(start_velocity=sv, end_velocity=ev)
        data = {tf.source: sample_trace}
        tf(data)
        out = data[tf.output].data

        assert len(out) < len(sample_trace.data), 'data should be trimmed'

    def test_params(self):
        with pytest.raises(TransformException):
            tf = TrimSurfaceWindow(start_velocity=int(100), end_velocity=5.0)

        with pytest.raises(TransformException):
            tf = TrimSurfaceWindow(start_velocity=5.0, end_velocity=int(100))

    def test_print(self):
        string = str(TrimSurfaceWindow(start_velocity=2.5, end_velocity=5.0))
        assert 'start_velocity' in string
        assert 'end_velocity' in string