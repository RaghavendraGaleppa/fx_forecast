import pytorch_lightning as pl
from . import tools

class ForexDataModule(pl.LightningDataModule):

    def prepare_data(self):
        self.window_size = 7
        self.batch_size = 128
        self.label_size = 1
        self.filenames=[
            # 'fx_forecast/data/DAT_MT_EURUSD_M1_2020.csv',
            'fx_forecast/data/DAT_MT_EURUSD_M1_202101.csv',  
            'fx_forecast/data/DAT_MT_EURUSD_M1_202102.csv',  
            'fx_forecast/data/DAT_MT_EURUSD_M1_202103.csv',  
            'fx_forecast/data/DAT_MT_EURUSD_M1_202104.csv',]

        price_data, price_labels = tools.build_dataset(
            *self.filenames,
            window_size=self.window_size,
            label_size=self.label_size
        )
        self.train_loader, self.val_loader = tools.split_and_create_loaders(price_data, price_labels, self.batch_size)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

        

