from . import TransformException, BaseTraceTransform
import numpy as np

class TargetLength(BaseTraceTransform):
    """
    trim or pad data to match a target input length

    Args:
        length: (int): the length of desired output
        random_offet: (bool): randomize the offset for trimming, false will start at beginning
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: normalized_length
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with filters applied
    """
    def __int__(
            self,
            length=1000,
            random_offset=False,
            source='raw',
            output='normalized_length',
            inplace=False):

        super().__init__(source, output, inplace)

        if not isinstance(length, int):
            raise TransformException(f'length must be a int, got {type(len())}')

        self.target_length = length
        self.random_offset = random_offset

    def __call__(self, data):
        super().__call__(data)

        normalized_length = data[self.source].copy()

        start_length = data.shape[-1]

        offset = 0
        if start_length > self.target_length:
            if self.random_offset:
                offset = np.random.randint(0, start_length - self.target_length)

        pad_length = max(self.target_length - start_length, 0)
        pad_tuple = [(0, 0) for k in range(len(normalized_length.shape))]
        pad_tuple[1] = (int(pad_length / 2), int(pad_length / 2) + (start_length % 2))
        normalized_length = np.pad(normalized_length, pad_tuple, mode='constant')
        normalized_length = normalized_length[:, offset:offset + self.target_length]

        return super().update(data, normalized_length)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'length: {self.length}, '
            f'random_offet: {self.random_offet})'
        )



