from . import TransformException
import obspy

class BaseTraceTransform:
    """
    base tranform for obspy Trace objects

    Args:
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: filtered
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(self, source='raw', output='filtered', inplace=False):
        self.source = source
        self.output = output
        self.inplace = inplace

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TransformException(f'data must be of type dict, got {type(data)}')

        if self.source not in data.keys():
            raise TransformException(f'source must be a key of data, got {data.keys()}', ', '.join(data.keys()))

        if not isinstance(data[self.source], obspy.Trace):
            raise TransformException(f'source must be of type obspy.Trace, got {type(data[self.source])}')

    def update(self, data_dict, tranformed_data):
        if self.inplace:
            data_dict[self.source] = tranformed_data
        else:
            data_dict[self.output] = tranformed_data

        return data_dict
