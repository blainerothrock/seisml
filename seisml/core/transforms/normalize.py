from . import TransformException, BaseTraceTransform
import numpy as np


class Normalize(BaseTraceTransform):
    """
    normalize seismic data using whiten transform (0 mean, 1 std)

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

        trace = data[self.source].copy()
        normalized = trace.data

        normalized -= normalized.mean()
        normalized /= normalized.std() + 1e-6

        trace.data = normalized

        return super().update(data, trace)

    def __repr__(self):
        return f'{self.__class__.__name__}()'