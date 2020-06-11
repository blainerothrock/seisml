import torch
import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    def __init__(self, num_layers=3, kernels=2, dims=(3, 8, 16, 16)):
        super(ConvAutoEncoder, self).__init__()

        self.num_layers = num_layers

        if isinstance(kernels, int):
            self.kernels = [kernels] * num_layers
        elif isinstance(kernels, tuple) and len(kernels) == num_layers:
            self.kernels = kernels
        else:
            raise ValueError('kernels must be a int or a tuple of size num_layers')

        if isinstance(dims, tuple) and len(dims) == num_layers + 1:
            self.dims = dims
        else:
            raise ValueError('dims must be tuple of size num_layers + 1')

        self.encoder = _ConvEncoder(
            num_layers=self.num_layers,
            kernels=self.kernels,
            dims=self.dims
        )

        self.decoder = _ConvDecoder(
            num_layers=self.num_layers,
            kernels=self.kernels,
            dims=self.dims
        )

    def forward(self, X):
        encoding, pool_indicies, sizes = self.encoder(X)
        X = self.decoder(encoding, pool_indicies, sizes)
        return X, encoding


class _ConvEncoder(nn.Module):

    def __init__(self, num_layers, kernels, dims):
        super(_ConvEncoder, self).__init__()

        self.num_layers = num_layers
        self.kernels = kernels
        self.dims = dims

        self.pool_idx = []

        self.convs = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.pools = nn.ModuleList()

        for l in range(num_layers):
            self.convs.append(
                nn.Conv1d(
                    self.dims[l],
                    self.dims[l + 1],
                    kernel_size=self.kernels[l]
                )
            )
            self.activations.append(nn.ReLU())
            self.pools.append(nn.MaxPool1d(2, return_indices=True))

    def forward(self, X):

        pool_indicies = []
        sizes = []
        for l in range(self.num_layers):
            X = self.convs[l](X)
            X = self.activations[l](X)
            sizes.append(X.size())
            X, pool_idx = self.pools[l](X)
            pool_indicies.append(pool_idx)

        return X, pool_indicies, sizes


class _ConvDecoder(nn.Module):

    def __init__(self, num_layers, kernels, dims):
        super(_ConvDecoder, self).__init__()

        self.num_layers = num_layers
        self.kernels = kernels
        self.dims = dims

        self.unpools = nn.ModuleList()
        self.transpose_convs = nn.ModuleList()
        self.activations = nn.ModuleList()

        for l in range(num_layers).__reversed__():
            self.unpools.append(nn.MaxUnpool1d(2))
            self.transpose_convs.append(
                nn.ConvTranspose1d(
                    self.dims[l + 1],
                    self.dims[l],
                    kernel_size=self.kernels[l]
                )
            )
            self.activations.append(nn.ReLU())

    def forward(self, X, pool_indices, sizes):
        pool_indices.reverse()
        sizes.reverse()

        for l in range(self.num_layers):
            X = self.unpools[l](
                X,
                pool_indices[l],
                output_size=sizes[l]
            )
            X = self.transpose_convs[l](X)
            X = self.activations[l](X)

        return X
