import pytorch_lightning as pl
from . import tools

class ForexDataModule(pl.LightningDataModule):

    def __init__(self, window_size, batch_size, label_size):
        self.window_size = window_size
        self.batch_size = batch_size
        self.label_size = label_size

    def prepare_data(self, filenames=[]):
        price_data, price_labels = tools.build_dataset(
            *filenames,
            window_size=self.window_size,
            batch_size=self.batch_size,
            label_size=self.label_size
        )
        self.train_loader, self.val_loader = tools.split_and_create_loaders(price_data, price_labels, BATCH_SIZE)

    def train_dataloader(self):
        return self.train_loader

    def val_loader(self):
        return self.val_loader

        

