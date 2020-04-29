from . import TransformException, BaseTraceTransform
import numpy as np


class Normalize(BaseTraceTransform):
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

    def __call__(self, data):
        super().__call__(data)

        normalized = data[self.source].copy()

        normalized -= data.mean()
        normalized /= data.std() + 1e-6

        return super().update(data, normalized)

    def __repr__(self):
        return f'{self.__class__.__name__}()'