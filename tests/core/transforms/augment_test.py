import pytest
import numpy as np
import obspy

from seisml.core.transforms import Augment, AugmentationType, TransformException


class TestAugment:

    def test_amplitude(self):
        np.random.seed(0)

        prob = 1.0
        src = 'src'
        out = 'out'

        tf = Augment(
            augmentation_types=[AugmentationType.AMPLITUDE],
            probability=1.0,
            source=src,
            output=out
        )

        trace = obspy.Trace(np.random.normal(0, 1, 1000000))
        orig_mean = trace.data.mean()
        orig_std = trace.data.std()

        data = {src: trace}
        tf(data)

        aug_mean = data[out].data.mean()
        aug_std = data[out].data.std()

        assert not np.allclose(data[src], data[out]), 'output should differ from input'
        assert np.allclose(orig_mean, aug_mean, atol=1e-3), 'output and input mean should be close'
        assert np.allclose(orig_std, aug_std, atol=1e-1), 'output and input mean should be close'

    def test_noise(self):
        np.random.seed(0)

        prob = 1.0
        src = 'src'
        out = 'out'

        tf = Augment(
            augmentation_types=[AugmentationType.NOISE],
            probability=1.0,
            source=src,
            output=out
        )

        trace = obspy.Trace(np.random.normal(0, 1, 1000000))
        orig_mean = trace.data.mean()
        orig_std = trace.data.std()

        data = {src: trace}
        tf(data)

        aug_mean = data[out].data.mean()
        aug_std = data[out].data.std()

        assert not np.allclose(data[src], data[out]), 'output should differ from input'
        assert np.allclose(orig_mean, aug_mean, atol=1e-2), 'output and input mean should be close'

    def test_params(self):
        with pytest.raises(TransformException) as e_info:
            tf = Augment(augmentation_types=0)
        with pytest.raises(TransformException) as e_info:
            tf = Augment(augmentation_types=[0, 1, 2])
        with pytest.raises(TransformException) as e_info:
            tf = Augment(probability='hello')
        with pytest.raises(TransformException) as e_info:
            tf = Augment(probability=1.01)

    def test_print(self):
        string = str(Augment())
        assert 'augmentation_type' in string