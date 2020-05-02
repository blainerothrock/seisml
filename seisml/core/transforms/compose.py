from . import TransformException, BaseTraceTransform


class Compose(BaseTraceTransform):
    """
    Composes several transforms together. Inspired by torchvision implementation.

    Args:
        transforms (list of ``BaseTraceTransfrom`` objects): list of transforms to compose.
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: transformed
        inplace: (Bool): optional, will overwrite the source data with the trim
    Example:
        >>> transforms.Compose([
        >>>     transforms.ButterworthPassFilter(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms, source='raw', output='transformed', inplace=False):
        super().__init__(source, output, inplace)
        self.transforms = transforms

    def __call__(self, data):
        super().__call__(data)

        _data = {self.source: data[self.source].copy()}

        for t in self.transforms:
            t.inplace = True
            t.source = self.source
            t(_data)

        return super().update(data, _data[self.source])

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
