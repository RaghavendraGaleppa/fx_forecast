import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score


class BasicLinearModel(pl.LightningModule):

    def __init__(self, window_size, label_size):
        super().__init__()

        self.layer_1 = torch.nn.Linear(window_size, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, label_size)

    def training_epoch_end(self, outputs):
        self.trainer.progress_bar_callback.main_progress_bar.write(
            f"Epoch {self.trainer.current_epoch} training loss={self.trainer.progress_bar_dict['loss']}")

    def validation_epoch_end(self, outputs):
        print(outputs)
        loss = outputs[0]
        loss = torch.stack(loss).mean()
        accuracy = outputs[1]
        self.trainer.progress_bar_callback.main_progress_bar.write(
            f"Epoch {self.trainer.current_epoch} validation_loss={loss.item()} validation_accuracy={accuracy}")

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
        return F.mse_loss(logits.view(-1), labels)

    def accuracy_metric(self, x, logits, labels):
        y_pred = x.view(x.shape[0],-1)[:,-1].view(-1) < logits.view(-1)
        y_true = x.view(x.shape[0],-1)[:,-1].view(-1) < labels.view(-1)
        return accuracy_score(y_true=y_true, y_pred=y_pred)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.mse_loss(logits, y)
        accuracy = self.accuracy_metric(x, logits, loss)
        self.log('train_loss', loss)
        self.log('accuracy_score', accuracy)
        return loss, accuracy

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.mse_loss(logits, y)
        accuracy = self.accuracy_metric(x, logits, loss)
        self.log('val_loss', loss)
        self.log('accuracy_score', accuracy)
        return loss, accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer




