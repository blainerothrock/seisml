from . import TransformException, BaseTraceTransform


class CustomTraceTransform(BaseTraceTransform):
    """
    Create any custom transform

    Arg:
        fn: (function): function to perform on data. Trace -> Trace
        identifier: (str): name of the transform, for reporting
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: transformed
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(
            self,
            fn,
            identifier,
            source='raw',
            output='transformed',
            inplace=False):
        super().__init__(source, output, inplace)

        self.fn = fn
        self.id = identifier

    def __call__(self, data):
        super().__call__(data)

        transformed = data[self.source].copy()
        transformed = self.fn(transformed)
        return super().update(data, transformed)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'id: {self.id})'
        )
