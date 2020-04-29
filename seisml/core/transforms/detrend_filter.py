from . import TransformException, BaseTraceTransform
import obspy
from enum import Enum

class DetrendType(str, Enum):
    SIMPLE = 'simple'
    LINEAR = 'linear'
    DEMEAN = 'demean'
    POLYNOMIAL = 'polynomial'
    SPLINE = 'spline'


class DetrendFilter(BaseTraceTransform):
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

        super().__init__(source, output, inplace)

        if not isinstance(detrend_type, DetrendType):
            raise TransformException(f'detrend_type must be a DetrendType, got {type(detrend_type)}')

        self.detrend_type = detrend_type

    def __call__(self, data):
        super().__call__(data)

        detrended = data[self.source].copy()
        detrended.detrend(type=self.detrend_type.value)

        return super().update(data, detrended)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'detrend_type: {self.detrend_type})'
        )