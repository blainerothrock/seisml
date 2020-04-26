from . import TransformException
import obspy
from enum import Enum

class DetrendType(str, Enum):
    SIMPLE = 'simple'
    LINEAR = 'linear'
    DEMEAN = 'demean'
    POLYNOMIAL = 'polynomial'
    SPLINE = 'spline'


class DetrendFilter:
    """
    remove a trend from seismic data

    Args:
        detrend_type: (DetrendType): detrend method [simple, linear, demean, polynomial, spline]
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: detrended
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with detrend applied
    """

    def __init__(
            self,
            detrend_type=DetrendType.DEMEAN,
            source='raw',
            output='detrended',
            inplace=False):

        if not isinstance(detrend_type, DetrendType):
            raise TransformException(f'detrend_type must be a DetrendType, got {type(detrend_type)}')

        self.detrend_type = detrend_type

        self.source = source
        self.output = output
        self.inplace = inplace

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TransformException(f'data must be of type dict, got {type(data)}')
        if self.source not in data.keys():
            raise TransformException(f'source must be a key of data, got {data.keys()}', ', '.join(data.keys()))\

        detrended = data[self.source].copy()
        detrended.detrend(type=self.detrend_type.value)

        if self.inplace:
            data[self.source] = detrended
        else:
            data[self.output] = detrended

        return data

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'detrend_type: {self.detrend_type})'
        )