from . import TransformException

class TrimSurfaceWindow:
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

        if not isinstance(start_velocity, float):
            raise TransformException(f'start_velocity must be a float, got {type(start_velocity)}')
        if not isinstance(end_velocity, float):
            raise TransformException(f'end_velocity must be a float, got {type(end_velocity)}')

        self.start_velocity = start_velocity
        self.end_velocity = end_velocity
        self.source = source
        self.output = output
        self.inplace = inplace

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TransformException(f'data must be of type dict, got {type(data)}')
        if self.source not in data.keys():
            raise TransformException(f'source must be a key of data, got {data.keys()}', ', '.join(data.keys()))

        trimmed = data[self.source].copy()

        start = int(trimmed.stats.sac['dist'] / self.start_velocity)
        end = int(trimmed.stats.sac['dist'] / self.end_velocity)
        t = trimmed.stats.starttime
        trimmed.trim(t + start, t + end, nearest_sample=False)

        if self.inplace:
            data[self.source] = trimmed
        else:
            data[self.output] = trimmed

        return data

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'start_velocity: {self.start_velocity}'
            f'end_velocity: {self.end_velocity}'
        )
