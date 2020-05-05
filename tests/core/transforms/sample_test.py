import pytest
from seisml.core.transforms import TransformException
from seisml.core.transforms.sample import Resample, DownSample, Interpolate

class TestSample:

    def test_resample(self, signal):
        data = {'raw': signal}
        new_rate = signal.stats.sampling_rate * 2
        tf = Resample(sampling_rate=new_rate)
        tf(data)
        out = data[tf.output]
        assert out.stats.sampling_rate == new_rate, 'new sampling rate should match'

    def test_downsample(self, signal):
        data = {'raw': signal}
        new_rate = signal.stats.sampling_rate / 2
        tf = DownSample(factor=2)
        tf(data)
        out = data[tf.output]
        assert out.stats.sampling_rate == new_rate, 'new sampling rate should match'

    def test_interpolate(self, signal):
        data = {'raw': signal}
        new_rate = signal.stats.sampling_rate / 1.5
        tf = Interpolate(sampling_rate=new_rate)
        tf(data)
        out = data[tf.output]
        assert out.stats.sampling_rate == new_rate, 'new sampling rate should match'

    def test_params(self):
        with pytest.raises(TransformException):
            tf = Resample(sampling_rate=40)

        with pytest.raises(TransformException):
            tf = Interpolate(sampling_rate=100)

        with pytest.raises(TransformException):
            tf = DownSample(factor=1.0)

    def test_print(self):
        string = str(Resample(sampling_rate=100.0))
        assert 'sampling_rate' in string

        string = str(Interpolate(sampling_rate=100.0))
        assert 'sampling_rate' in string

        string = str(DownSample(factor=2))
        assert 'factor' in string