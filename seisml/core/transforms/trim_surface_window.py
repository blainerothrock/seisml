from . import TransformException, BaseTransform

class TrimSurfaceWindow(BaseTransform):
    """
    simple surface window trim based on velocity

    Args:
        start_velocity: (float): velocity to indicate the start of the window
        end_velocity: (float): velocity to indicate the end of the window
        source: (string): optional, the data source to apply the trim, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: trimmed
        inplace: (Bool): optional, will overrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with surface window trimmed
    """
    def __init__(
            self,
            start_velocity,
            end_velocity,
            source='raw',
            output='trimmed',
            inplace=False):

        super().__init__(source, output, inplace)

        if not isinstance(start_velocity, float):
            raise TransformException(f'start_velocity must be a float, got {type(start_velocity)}')
        if not isinstance(end_velocity, float):
            raise TransformException(f'end_velocity must be a float, got {type(end_velocity)}')

        self.start_velocity = start_velocity
        self.end_velocity = end_velocity

    def __call__(self, data):
        super().__call__(data)

        trimmed = data[self.source].copy()

        start = int(trimmed.stats.sac['dist'] / self.start_velocity)
        end = int(trimmed.stats.sac['dist'] / self.end_velocity)
        t = trimmed.stats.starttime
        trimmed.trim(t + start, t + end, nearest_sample=False)

        return super().update(data, trimmed)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'start_velocity: {self.start_velocity}'
            f'end_velocity: {self.end_velocity}'
        )
