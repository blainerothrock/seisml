from . import TransformException, BaseTransform
from enum import Enum
import numpy as np

class AugmentationType(Enum):
    AMPLITUDE = 0
    NOISE = 1

class Augment(BaseTransform):
    """
    randomly add amplitude or noise to data

    Args:
        augmentation_types: (list:AugmentationType): what augmentations to perform, all will be performed in order
        probability: (float): the random chance of augmentation, default: 0.5
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: augmented
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException: if a filter in not in a supporting list

    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(
            self,
            augmentation_types=[AugmentationType.AMPLITUDE, AugmentationType.NOISE],
            probability=0.5,
            source='raw',
            output='augmented',
            inplace=False):

        super().__init__(source, output, inplace)

        if not isinstance(augmentation_types, list):
            raise TransformException(f'augmentation_types must be a list, got {type(AugmentationType)}')

        for t in augmentation_types:
            if not isinstance(t, AugmentationType):
                raise TransformException(f'augmentation type must be a AugmentationType, got {type(t)}')

        if not isinstance(probability, float):
            raise TransformException(f'probability must be a float, got {type(probability)}')
        if probability > 1.0 or probability < 0.0:
            raise TransformException(f'probability must be between 0.0 and 1.0, got {probability}')

        self.augmentation_type = augmentation_types
        self.probability = probability

    def __call__(self, data):
        super().__call__(data)

        augmented = data[self.source].copy()

        if np.random.random() < self.probability:
            if AugmentationType.AMPLITUDE in self.augmentation_type:
                start_gain, end_gain = [np.random.uniform(0, 2), np.random.uniform(0, 2)]
                amplitude_mod = np.linspace(start_gain, end_gain, num=augmented.data.shape[-1])
                augmented.data *= amplitude_mod

            if AugmentationType.NOISE in self.augmentation_type:
                std = data[self.source].data.std() * np.random.uniform(1, 2)
                mean = data[self.source].data.mean()
                noise = np.random.normal(loc=mean, scale=std, size=augmented.data.shape)
                augmented.data += noise

        return super().update(data, augmented)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'augmentation_type: {self.augmentation_type}, '
            f'probability: {self.probability}, '
        )