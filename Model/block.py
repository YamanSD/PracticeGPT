import torch
import torch.nn as nn

from .multiHeadAttention import MultiHeadAttention
from .feedForward import FeedForward


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int) -> None:
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()

        head_size: int = n_embd // n_head
        self.sa: MultiHeadAttention = MultiHeadAttention(n_head, head_size)
        self.ffwd: FeedForward = FeedForward(n_embd)
        self.ln1: nn.LayerNorm = nn.LayerNorm(n_embd)
        self.ln2: nn.LayerNorm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.tensor) -> torch.tensor:
        x += self.sa(self.ln1(x))
        x += self.ffwd(self.ln2(x))

        return x
