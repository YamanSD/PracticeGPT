import torch
import torch.nn as nn

from constants import *


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int) -> None:
        super().__init__()

        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.net(x)
