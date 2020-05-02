import pytest
import os
from seisml.utility.download_data import download_sample_data
from seisml.datasets.triggered_earthquake import TriggeredEarthquake

class TestTriggeredEarthquake:

    def test_download_and_preproces(self):
        ds = TriggeredEarthquake(folder='~/.seisml/data/sample_data',
                                 force_download=False,
                                 download=download_sample_data)

        assert os.path.isdir(os.path.expanduser('~/.seisml/data/sample_data/prepared')), 'prepared dir should exist'
