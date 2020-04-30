import pytest


@pytest.fixture
def sample_trace():
    import obspy

    s = obspy.read('tests/sample_data/sample.SAC')
    return s[0]
