from seisml.core.transforms import TransformException, BaseTraceTransform

class Interpolate(BaseTraceTransform):
    """
    reasmple using Fourier method, passthrough of obspy.core.trace.Trace.interpolate

    Args:
        sampling_rate: (float): factor to down sample by
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: interpolated
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(
            self,
            sampling_rate,
            source='raw',
            output='interpolated',
            inplace=False):

        super().__init__(source, output, inplace)

        if not isinstance(sampling_rate, float):
            raise TransformException(f'sampling_rate must be a float, got {type(sampling_rate)}')

        self.sampling_rate = sampling_rate

    def __call__(self, data):
        super().__call__(data)

        down_sampled = data[self.source].copy().interpolate(self.sampling_rate)
        return super().update(data, down_sampled)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'sampling_rate: {self.sampling_rate})'
        )