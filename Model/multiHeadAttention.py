import torch
import torch.nn as nn

from .head import Head
from constants import *


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads: int, head_size: int) -> None:
        super().__init__()
        self.heads: nn.ModuleList[Head] = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj: torch.tensor = nn.Linear(n_embed, n_embed)
        self.dropout: torch.tensor = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out: torch.tensor = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))
