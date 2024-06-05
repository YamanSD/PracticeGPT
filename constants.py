# Batch
B: int = 4

# Time
T: int = 8

# Channel
C: int = 32

# Number of independent sequences to be processed in parallel
batch_size: int = B

# Maximum context length of the predictions
block_size: int = T

# Seed to control the randomness
seed: int = 1337

# Number of evaluation iterations
eval_iters: int = 200

# Max training iteration
max_iters: int = 10_000

# Evaluation intervals
eval_interval = 100

# Number of embeddings
n_embed: int = C

# Size for the head networks
n_head: int = 16

# Depth of the network
n_layer: int = 4

# Dropout rate
dropout: float = 0

# Learning rate of the network
learning_rate: float = 1e-3
