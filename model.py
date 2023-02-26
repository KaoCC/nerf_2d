from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class Nerf2DMLP(pl.LightningModule):
    def __init__(self, n_inputs, n_hidden, n_outputs):
        super().__init__()

        # encoder ?

        # layers
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
            nn.ReLU(),
        )

        # activation ?

    def forward(self, x):
        return self.layers(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_index):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
