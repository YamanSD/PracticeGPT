from typing import Literal

import torch.nn as nn
from torch import tensor, multinomial, cat, no_grad, zeros, arange

from constants import *
from Data import get_batch
from Tokenizer import vocab_size
from .block import Block


class BigramLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table: nn.Embedding = (
            nn.Embedding(
                vocab_size,
                n_embed
            )
        )
        self.blocks: tensor = nn.Sequential(*(Block(n_embed, n_head=n_head) for _ in range(n_layer)))
        self.ln_f: tensor = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head: tensor = nn.Linear(n_embed, vocab_size)

    @no_grad()
    def estimate_loss(self) -> dict[str, tensor]:
        out: dict[str, tensor] = {}
        self.eval()

        split: Literal['train', 'test']
        for split in ('train', 'test'):
            losses: tensor = zeros(eval_iters)

            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = self(X, Y)
                losses[X] = loss.item()
            out[split] = losses.mean()

        self.train()
        return out

    def forward(self, idx: tensor, targets: tensor = None) -> tuple[tensor, tensor]:
        B, T = idx.shape

        """
        (B, T, C) B is the batch size, T is time (block size), and C is the channel (vocab size)
        ids and targets are both (B, T) tensor of integers

        :param idx: (B, T) tensor.
        :param targets: (B, T) tensor.
        :return: Tensor of shape (B, T, C).
        """
        # idx and targets are both (B,T) tensor of integers
        tok_emb: tensor = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb: tensor = self.position_embedding_table(arange(T))  # (T,C)
        x: tensor = tok_emb + pos_emb  # (B,T,C)
        x: tensor = self.blocks(x)  # (B,T,C)
        x: tensor = self.ln_f(x)  # (B,T,C)
        logits: tensor = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits: tensor = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss: tensor = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx: tensor, max_new_tokens: int) -> tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)

            # focus only on the last time step
            logits: tensor = logits[:, -1, :]  # Becomes (B, C)

            probs: tensor = nn.functional.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next: multinomial = multinomial(probs, num_samples=1)  # (B, 1)

            # append samples index to the running sequence
            idx = cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx
