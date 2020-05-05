from seisml.core.transforms import TransformException, BaseTraceTransform

class DownSample(BaseTraceTransform):
    """
    reasmple using Fourier method, passthrough of obspy.core.trace.Trace.decimate

    Args:
        factor: (int): factor to down sample by
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: downsampled
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(
            self,
            factor,
            source='raw',
            output='downsampled',
            inplace=False):

        super().__init__(source, output, inplace)

        if not isinstance(factor, int):
            raise TransformException(f'sampling_rate must be a int, got {type(factor)}')

        self.factor = factor

    def __call__(self, data):
        super().__call__(data)

        down_sampled = data[self.source].copy().decimate(self.factor)
        return super().update(data, down_sampled)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'factor: {self.factor})'
        )