import pytest
import numpy as np
import obspy


@pytest.fixture
def sample_trace():
    import obspy

    s = obspy.read('tests/sample_data/sample.SAC')
    return s[0]

@pytest.fixture
def signal():
    np.random.seed(1234)
    time_step = 0.02
    period = 5.
    time_vec = np.arange(0, 20, time_step)
    sig = (np.sin(2 * np.pi / period * time_vec)
           + 0.5 * np.random.randn(time_vec.size))
    return obspy.Trace(sig)

@pytest.fixture
def signal_with_linear():
    np.random.seed(1234)
    t = np.linspace(0, 5, 100)
    x = t + np.random.normal(size=100)

    return obspy.Trace(x)
