from torch import nn
import torch
import lightning as pl
import torch.nn.functional as F
import math


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
