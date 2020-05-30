import torch
from torch import nn


class DeepClusteringLoss(nn.Module):
    def __init__(self):
        """
        Computes the deep clustering loss with weights. Equation (7) in [1].

        References:
            [1] Wang, Z. Q., Le Roux, J., & Hershey, J. R. (2018, April).
            Alternative Objective Functions for Deep Clustering.
            In Proc. IEEE International Conference on Acoustics,  Speech
            and Signal Processing (ICASSP).
        """
        super(DeepClusteringLoss, self).__init__()

    def forward(self, embedding, assignments, weights=None):
        batch_size = embedding.shape[0]
        embedding_size = embedding.shape[-1]
        num_sources = assignments.shape[-1]

        embedding = embedding.view(batch_size, -1, embedding_size)
        assignments = assignments.view(batch_size, -1, num_sources)
        num_points = embedding.shape[1]

        if weights is None:
            weights = embedding.new(batch_size, num_points).fill_(1.0)
        weights = weights.view(batch_size, -1, 1)
        assignments = weights.expand_as(assignments) * assignments
        embedding = weights.expand_as(embedding) * embedding
        norm = ((((weights) ** 2)).sum(dim=1) ** 2).sum()

        vTv = ((embedding.transpose(2, 1) @ embedding) ** 2).sum()
        vTy = ((embedding.transpose(2, 1) @ assignments) ** 2).sum()
        yTy = ((assignments.transpose(2, 1) @ assignments) ** 2).sum()
        loss = (vTv - 2 * vTy + yTy) / norm.detach()
        return loss


class WhitenedKMeansLoss(nn.Module):
    """
    Computes the whitened K-Means loss with weights. Equation (6) in [1].
    References:
    [1] Wang, Z. Q., Le Roux, J., & Hershey, J. R. (2018, April).
        Alternative Objective Functions for Deep Clustering.
        In Proc. IEEE International Conference on Acoustics,  Speech
        and Signal Processing (ICASSP).
    """
    DEFAULT_KEYS = {
        'embedding': 'embedding',
        'ideal_binary_mask': 'assignments',
        'weights': 'weights'
    }

    def __init__(self):
        super(WhitenedKMeansLoss, self).__init__()

    def forward(self, embedding, assignments, weights):
        batch_size = embedding.shape[0]
        embedding_size = embedding.shape[-1]
        num_sources = assignments.shape[-1]
        weights = weights.view(batch_size, -1, 1)
        # make everything unit norm
        embedding = embedding.reshape(batch_size, -1, embedding_size)
        embedding = nn.functional.normalize(embedding, dim=-1, p=2)
        assignments = assignments.view(batch_size, -1, num_sources)
        assignments = nn.functional.normalize(assignments, dim=-1, p=2)
        assignments = weights * assignments
        embedding = weights * embedding
        embedding_dim_identity = torch.eye(
            embedding_size, device=embedding.device).float()
        source_dim_identity = torch.eye(
            num_sources, device=embedding.device).float()
        vTv = (embedding.transpose(2, 1) @ embedding)
        vTy = (embedding.transpose(2, 1) @ assignments)
        yTy = (assignments.transpose(2, 1) @ assignments)
        ivTv = torch.inverse(vTv + embedding_dim_identity)
        iyTy = torch.inverse(yTy + source_dim_identity)
        ivTv_vTy = ivTv @ vTy
        vTy_iyTy = vTy @ iyTy
        # tr(AB) = sum(A^T o B)
        # where o denotes element-wise product
        # this is the trace trick
        # http://andreweckford.blogspot.com/2009/09/trace-tricks.html
        trace = (ivTv_vTy * vTy_iyTy).sum()
        D = (embedding_size + num_sources) * batch_size
        loss = D - 2 * trace
        return loss / batch_size
