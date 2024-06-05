from torch import tensor, long as t_long, manual_seed, stack, randint
from typing import Literal

from constants import *
from Tokenizer import text, encode

# Data tensor
data: tensor = tensor(encode(text), dtype=t_long)

# Split index of train-test data
n: int = int(0.9 * len(data))

# Train-test datasets
train_data: tensor = data[:n]
val_data: tensor = data[n:]

# To keep the randomness consistent
manual_seed(seed)


def get_batch(split: Literal["train", "test"]) -> tuple[stack, stack]:
    """
    :param split: Type of the split.
    :return: x, y batch data.
    """
    d: tensor = train_data if split == "train" else val_data
    ix: tensor = randint(len(d) - block_size, (batch_size,))

    x: stack = stack([d[i: i + block_size] for i in ix])
    y: stack = stack([d[i + 1: i + block_size + 1] for i in ix])
    return x, y
