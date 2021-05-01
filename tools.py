import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

def load_csv(filename, columns=['date', 'time', 'start','high','low','end','UNK']):
    df = pd.read_csv(filename)
    if columns:
        df.columns = columns

    return df

def create_dataset_custom_scaler(series, window_size=5, hop_size=1, label_size=1):
    series = series.iloc[list(range(0,len(series),hop_size))]
    new_data_df = pd.DataFrame(series)
    new_data_df['labels'] = new_data_df.shift(-window_size)
    price_data = []
    price_labels = []
    for i in tqdm(range(0, len(new_data_df)-window_size-1)):
        min_scale = np.random.uniform(0.05, 0.2, size=(1,))[0]
        max_scale = np.random.uniform(0.8, 0.95, size=(1,))[0]
        custom_feature_range = (min_scale, max_scale)
        scaler = MinMaxScaler(feature_range=custom_feature_range)
        prices = new_data_df.start.iloc[i:i+window_size+1].values.reshape(-1)
        prices = scaler.fit_transform(prices.reshape(-1,1))
        if label_size == 1:
            next_price = prices.reshape(-1)[-1]
        else:
            next_price = new_data_df.labels.iloc[i:i+label_size]
        price_data.append(prices[:-1])
        price_labels.append(next_price)

    return np.array(price_data), np.array(price_labels)


def build_dataset(*filenames, **kwargs):
    price_data_list = []
    price_labels_list = []
    for filename in filenames:
        data_df = load_csv(filename, columns=['date', 'time', 'start','high','low','end','UNK'])
        price_data , price_labels = create_dataset_custom_scaler(
                    series=data_df.start,
                    window_size=kwargs.get('window_size', 7),
                    hop_size=kwargs.get('hop_size',1),
                    label_size=kwargs.get('label_size',1)
                )
        price_data_list.append(price_data)
        price_labels_list.append(price_labels)

    return np.vstack(price_data_list), np.hstack(price_labels_list)

def split_train_test(price_data, price_labels, split_pct=0.9):
    total_split = int(len(price_data)*split_pct)
    print(f"Using split_pct of {split_pct}")

    train_price_data = price_data[:total_split]
    train_price_labels = price_labels[:total_split]

    test_price_data = price_data[total_split:]
    test_price_labels = price_labels[total_split:]

    print(f"Train data shape: {train_price_data.shape}, {train_price_labels.shape}")
    print(f"Test data shape: {test_price_data.shape}, {test_price_labels.shape}")

    return (train_price_data, train_price_labels), (test_price_data, test_price_labels)


class ForexDataset(Dataset):

    def __init__(self, price_data, price_labels):
        self.price_data = price_data.copy().transpose(0,2,1)
        self.price_labels = price_labels.copy()

    def __len__(self):
        return len(self.price_data)

    def __getitem__(self, idx):
        return self.price_data[idx], self.price_labels[idx]

def split_and_create_loaders(price_data, price_labels, batch_size=128):
    
    (train_price_data, train_price_labels), (test_price_data, test_price_labels) = split_train_test(price_data, price_labels)
    train_dataset = ForexDataset(train_price_data, train_price_labels)
    test_dataset = ForexDataset(test_price_data, test_price_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
