import os
from pathlib import Path

__version__ = 0.1

from . import datasets
from . import metrics
from . import networks
from . import utility

DATA_PATH = os.path.expanduser('~/.seisml/data/')
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)