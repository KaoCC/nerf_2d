from torch import nn
import torch
import lightning as pl
import torch.nn.functional as F
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, n_input_dim, n_hidden_dim, n_output_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_input_dim, n_hidden_dim),
            nn.ReLU(),
            nn.Linear(n_hidden_dim, n_hidden_dim),
            nn.ReLU(),
            nn.Linear(n_hidden_dim, n_hidden_dim),
            nn.ReLU(),
            nn.Linear(n_hidden_dim, n_output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


# Fourier feature mapping
class Frequency(nn.Module):
    def __init__(self, input_dim: int, n_frequencies: int = 7):

        # Given a scalar value x in [-1, 1], transform it to a vector:
        # [sin(pi x), cos( pi x), sin(2 pi x), cos(2 pi x), ... , sin(2^(n-1) pi x), cos(2^(n-1) pi x)]

        super().__init__()

        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.output_dim = self.input_dim * self.n_frequencies * 2

        freqs = math.pi * (2.0 ** torch.linspace(0.0, n_frequencies - 1, n_frequencies))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor):

        x = x.unsqueeze(dim=-1)
        x = x * self.freqs
        x = torch.cat((torch.sin(x), torch.cos(x)), dim=-1)
        return x.flatten(-2, -1)


class Nerf2DMLP(pl.LightningModule):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()

        # encoder
        self.encoder = Frequency(n_inputs)

        # MLP
        self.mlp = MLP(self.encoder.output_dim, n_hidden, n_outputs)

    def forward(self, x):

        # input should be normalized to be in [-1, 1]

        # encode
        enc_x = self.encoder(x)
        result = self.mlp(enc_x)

        # result is in [0, 1]
        # scale to be in [0, 255]
        return result * 255

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss



PRIMES = [
    73856093,
    19349663,
    83492791,
    49979687,
    97755331,
    13623469,
    31912469,
    2654435761,
]

def hash_func_simple(indices, primes, hashmap_size):
    d = indices.shape[-1]
    hashed = ((indices * primes[:d]).sum(dim=-1)) % hashmap_size
    return hashed

@torch.no_grad()
def hash_func(indices: torch.Tensor, primes: torch.Tensor, hashmap_size: int):

    """
    indices: (..., D) integer tensor of grid indices
    primes:  (D,) tensor of large primes
    hashmap_size: size of the hash table (int, typically power of 2)

    Returns:
        hashed_indices: (...,) tensor of hashed indices into the table
    """

    assert indices.dtype == torch.int64

    # neighbors
    d = indices.shape[-1]

    hashed = indices[..., 0] * primes[0]
    for i in range(1, d):
        hashed ^= indices[..., i] * primes[i]
    return hashed % hashmap_size



class Grid(nn.Module):
    def __init__(
        self, input_dim: int, n_features: int, hashmap_size: int, resolution: float
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.hashmap_size = hashmap_size
        self.resolution = resolution

        self.embedding = nn.Embedding(hashmap_size, n_features)
        nn.init.uniform_(self.embedding.weight, a=-1e-4, b=1e-4)

        # for hash
        primes = torch.tensor(PRIMES, dtype=torch.int64)
        self.register_buffer("primes", primes, persistent=False)

        n_neighbors = 1 << self.input_dim
        neighbors = np.arange(n_neighbors, dtype=np.int64).reshape((-1, 1))
        dims = np.arange(self.input_dim, dtype=np.int64).reshape((1, -1))

        # binary mask for interpolation
        binary_mask_np = (neighbors & (1 << dims)) == 0
        binary_mask = torch.tensor(binary_mask_np, dtype=bool)
        self.register_buffer("binary_mask", binary_mask, persistent=False)

    def forward(self, x: torch.Tensor):

        # x:  (batch_size, input_dim)

        # transform each element from [-1, 1] t0 [0, 1]
        x = x + 1
        x = x / 2

        batch_dims = len(x.shape[:-1])

        # print(batch_dims)

        x = x * self.resolution

        x_i = x.long()
        x_f = x - x_i.float().detach()

        # (batch_size, 1, input_dim)
        x_i = x_i.unsqueeze(dim=-2)
        x_f = x_f.unsqueeze(dim=-2)

        # (1, n_neighbors, input_dim)
        binary_mask = self.binary_mask.reshape(
            (1,) * batch_dims + self.binary_mask.shape
        )

        # print(binary_mask)
        # print(binary_mask.shape)

        # (batch_size, n_neighbors, input_dim)
        indices = torch.where(binary_mask, x_i, x_i + 1)
        weights = torch.where(binary_mask, 1 - x_f, x_f)

        weight = weights.prod(dim=-1, keepdim=True)

        hash_ids = hash_func(indices, self.primes, self.hashmap_size)

        neighbor_data = self.embedding(hash_ids)
        return torch.sum(neighbor_data * weight, dim=-2)


class Nerf2DGridMLP(pl.LightningModule):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()

        # Grid
        # Input -> N-D feature vector

        self.encoder = Grid(n_inputs, 3, 2**17, 256)

        # MLP
        self.mlp = MLP(self.encoder.n_features, n_hidden, n_outputs)

    def forward(self, x: torch.Tensor):

        # encode
        enc_x = self.encoder(x)
        result = self.mlp(enc_x)

        return result * 255

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
