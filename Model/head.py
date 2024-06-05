import torch
import torch.nn as nn
from torch.nn import functional
from math import sqrt

from constants import *


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key: torch.tensor = nn.Linear(n_embed, head_size, bias=False)
        self.query: torch.tensor = nn.Linear(n_embed, head_size, bias=False)
        self.value: torch.tensor = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, C = x.shape
        k: torch.tensor = self.key(x)  # (B,T,C)
        q: torch.tensor = self.query(x)  # (B,T,C)

        # compute attention scores ("affinities")
        wei: torch.tensor = q @ k.transpose(-2, -1) / sqrt(C)  # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = functional.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v: torch.tensor = self.value(x)  # (B,T,C)

        return wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
