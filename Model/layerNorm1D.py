from torch import ones, zeros, sqrt, tensor


class LayerNorm1d:
    """
    Class representing the normalization part of the Norm & Add layer.
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps: float = eps
        self.gamma: tensor = ones(dim)
        self.beta: tensor = zeros(dim)

    def __call__(self, x: tensor) -> tensor:
        # calculate the forward pass
        xmean: tensor = x.mean(1, keepdim=True)  # batch mean
        xvar: tensor = x.var(1, keepdim=True)  # batch variance
        xhat: tensor = (x - xmean) / sqrt(xvar + self.eps)  # normalize to unit variance
        self.out: tensor = self.gamma * xhat + self.beta

        return self.out

    def parameters(self) -> list[tensor]:
        return [self.gamma, self.beta]
