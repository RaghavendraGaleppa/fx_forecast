import torch
from torch.nn import functional as F
import pytorch_lightning as pl


class BasicLinerModel(pl.LightningModule):

    def __init__(self, window_size, label_size):
        super().__init__()

        self.layer_1 = torch.nn.Linear(window_size, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, label_size)

    def forward(self, x):
        batch_size, window_size, features = x.size()

        x = x.view(batch_size, -1)

        x = self.layer_1(x)
        x = torch.relu(x)

        x = self.layer_2(x)
        x = torch.relu(x)

        x = self.layer_3(x)

        return x

    def mse_loss(self, logits, labels):
        return F.mse_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.mse_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.mse_loss(logits, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




