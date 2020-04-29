import obspy
from . import TransformException, BaseTraceTransform
from enum import Enum

class FilterType(str, Enum):
    BANDPASS = 'bandpass'
    BANDSTOP = 'bandstop'
    HIGHPASS = 'highpass'
    LOWPASS = 'lowpass'


class ButterworthPassFilter(BaseTraceTransform):
    """
    standard seismic filtering on a Stream or Trace object

    Args:
        filter_type: (FilterType): type of filter
        min_freq (float): minimum corner frequency for the pass/stop (not used in lowpass)
        max_freq (float): maximum corner frequency for the pass/stop (not used in highpass)
        corners (int): filter corners/order
        zerophase (bool): If True, apply filter once forwards and once backwards. This results in twice the filter order
         but zero phase shift in the resulting filtered trace.
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: filtered
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException: if a filter in not in a supporting list

    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(
            self,
            filter_type='bandpass',
            min_freq=0.0,
            max_freq=10.0,
            corners=2,
            zerophase=False,
            source='raw',
            output='filtered',
            inplace=False):

        super().__init__(source, output, inplace)

        self._filters = ['highpass', 'lowpass', 'bandpass', 'bandstop']

        if not isinstance(filter_type, FilterType):
            raise TransformException(f'filter_type must be a FilterType, got {type(filter_type)}')
        if not isinstance(min_freq, float):
            raise TransformException(f'min_freq must be a float, got {type(min_freq)}')
        if not isinstance(max_freq, float):
            raise TransformException(f'max_freq must be a float, got {type(max_freq)}')
        if not isinstance(corners, int):
            raise TransformException(f'corners must be a int, got {type(corners)}')
        if not isinstance(zerophase, bool):
            raise TransformException(f'zerophase must be a bool, got {type(zerophase)}')

        self.filter_type = filter_type
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.corners = corners
        self.zerophase = zerophase


    def __call__(self, data):
        super().__call__(data)

        transformed = data[self.source].copy()

        if self.filter_type in [FilterType.BANDPASS, FilterType.BANDSTOP]:
            transformed.filter(
                self.filter_type.value,
                freqmin=self.min_freq,
                freqmax=self.max_freq,
                corners=self.corners,
                zerophase=self.zerophase
            )
        elif self.filter_type in [FilterType.HIGHPASS, FilterType.LOWPASS]:
            transformed.filter(
                self.filter_type.value,
                freq=self.min_freq if self.filter_type == FilterType.HIGHPASS else self.max_freq,
                corners=self.corners,
                zerophase=self.zerophase
            )

        return super().update(data, transformed)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'filter_type: {self.filter_type}, '
            f'min_freq: {self.min_freq}, '
            f'max_freq: {self.max_freq}, '
            f'corners: {self.corners}, '
            f'zerophase: {self.zerophase})'
        )
