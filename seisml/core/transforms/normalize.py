from . import TransformException
import numpy as np


class Normalize:
    """
    normalize seismic data using whiten transform

    Args:
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: normalized
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns
        data: a modified dictionary with filters applied
    """

    def __init__(
            self,
            source='raw',
            output='normalized',
            inplace=False):

        self.source = source
        self.output = output
        self.inplace = inplace

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TransformException(f'data must be of type dict, got {type(data)}')
        if self.source not in data.keys():
            raise TransformException(f'source must be a key of data, got {data.keys()}', ', '.join(data.keys()))

        normalized = data[self.source].copy()

        normalized -= data.mean()
        normalized /= data.std() + 1e-6

        if self.inplace:
            data[self.source] = normalized
        else:
            data[self.output] = normalized

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'