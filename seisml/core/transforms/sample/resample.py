from seisml.core.transforms import TransformException, BaseTraceTransform

class Resample(BaseTraceTransform):
    """
    reasmple using Fourier method, passthrough of obspy.core.trace.Trace.reample

    Args:
        sampling_rate: (float): new sample rate in Hz.
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: resampled
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
            output='resampled',
            inplace=False):

        super().__init__(source, output, inplace)

        if not isinstance(sampling_rate, float):
            raise TransformException(f'sampling_rate must be a float, got {type(sampling_rate)}')

        self.sampling_rate = sampling_rate

    def __call__(self, data):
        super().__call__(data)

        resampled = data[self.source].copy().resample(self.sampling_rate)
        return super().update(data, resampled)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'sampling_rate: {self.sampling_rate})'
        )

