import obspy
import torch
from . import TransformException, BaseTraceTransform

class ToTensor(BaseTraceTransform):
    """
    convert obspy Trace to Tensor

    Args:
        device (torch.device)
        source (string): the data source to filter, default: raw
        output: (string): optional, the key of the output data in the dictionary, default: tensor
        inplace: (Bool): optional, will overwrite the source data with the trim

    Raises:
        TransformException

    Returns:
        data: a modified dictionary with filters applied
    """

    def __init__(self, device, source='raw', output='tensor', inplace=False):
        super().__init__(source, output, inplace)

        if not isinstance(device, torch.device):
            raise TransformException(f'device must be a torch.device, got {type(device)}')

        self.device = device

    def __call__(self, data):
        super().__call__(data)

        tensor = data[self.source].copy()

        tensor = torch.tensor(tensor.data, dtype=torch.float32, device=self.device)

        super().update(data, tensor)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'device: {self.device})'
        )

