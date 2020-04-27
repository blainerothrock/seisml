from . import TransformException

class BaseTransform:
    def __init__(self, source='raw', output='filtered', inplace=False):
        self.source = source
        self.output = output
        self.inplace = inplace

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TransformException(f'data must be of type dict, got {type(data)}')

        if self.source not in data.keys():
            raise TransformException(f'source must be a key of data, got {data.keys()}', ', '.join(data.keys()))

        # TODO: add check for data type (Source or Trace)

    def update(self, data_dict, tranformed_data):
        if self.inplace:
            data_dict[self.source] = tranformed_data
        else:
            data_dict[self.output] = tranformed_data

        return data_dict
