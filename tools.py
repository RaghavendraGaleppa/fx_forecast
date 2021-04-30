import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

def load_csv(filename, columns=None):
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
        prices = new_data_df.start.iloc[i:i+window_size+1].values
        prices = scaler.fit_transform(prices.reshape(-1,1))
        if label_size == 1:
            next_price = prices[-1][0]
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
                    data_df.start,
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

    return (train_price_data, train_price_labels_), (test_price_data, test_price_labels)


