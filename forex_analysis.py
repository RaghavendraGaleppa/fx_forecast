import requests
import torch
import numpy as np
import pandas as pd
from .models import load_model_from_checkpoint

from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tqdm import tqdm
import requests

from collections import deque
import time
import logging
import sys

from .tools import load_csv


def get_eur_usd_price_data():
    url = 'https://webrates.truefx.com/rates/connect.html?f=html'
    data = requests.get(url)
    split_data = data.text.split('<tr>')[1].replace('<td>','').split('</td>')
    price_data = {
        'timestamp': int(split_data[1][:-3]),
        'price_a': float(split_data[2]+split_data[3]),
        'price_b': float(split_data[4]+split_data[5]),
    }
    price_data['final_price'] = (price_data['price_a'] + price_data['price_b'])/2
    return price_data

class ForexDataSimulation():

    def __init__(self, path_to_csv, window_size, label_size):
        self.df = load_csv(path_to_csv)
        self.window_size = window_size
        self.label_size = label_size
        self.idx = 0

    def reset(self):
        self.idx = 0
        
    def get_next_price(self):

        data = {'timestamp': int(time.time()), 'final_price': self.df.start.iloc[self.idx]}
        self.idx += 1
        return data
     
class DataStream():
    def __init__(self, model_checkpoint_path, simulation=None):


        self.model, self.window_size, self.label_size, self.model_type = load_model_from_checkpoint(
            checkpoint_path=model_checkpoint_path,
        )

        self.simulation = simulation

        """ variables for data acquiring and storing """
        self.raw_data_queue = deque(maxlen=self.window_size)
        self.normalized_data = []
        self.fx_live_api = 'https://webrates.truefx.com/rates/connect.html?f=html'
        self.last_timestamp = None

        """ a flag which says whether the data has been update or not """
        self.data_updated_flag = 0 # 0 means not updated yet and 1 means data updated

        """ setup the logger """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        sys.stdout.flush()
        if not self.logger.handlers:
            stdout_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            stdout_handler.setFormatter(formatter)
            self.logger.addHandler(stdout_handler)

        """ model predictions """
        self.predicted_prices = []
        self.actual_prices = []


    def update_prices_data(self):
        if self.simulation:
            price_data = self.simulation.get_next_price()
        else:
            price_data = get_eur_usd_price_data()

        if self.last_timestamp == price_data['timestamp']:
            self.logger.debug("Price not updated in the API")
            return

        """ Update the latest timestamp """
        self.last_timestamp = price_data['timestamp']

        self.raw_data_queue.append(price_data['final_price'])

        """ Log information """
        self.logger.debug(f"Timestamp:{self.last_timestamp}, Price: {price_data['final_price']}"
                            f", queue length: {len(self.raw_data_queue)}")

        if len(self.raw_data_queue) == self.window_size:

            """ Normalize the prices for next price prediction """
            data_ndarray = np.array(self.raw_data_queue)

            """ Create a scaler and fit the raw data """
            scaler = MinMaxScaler(feature_range=(0.1,0.9))
            self.normalized_data = scaler.fit_transform(data_ndarray.reshape(-1,1)) 
            if self.model_type == 'torch':
                self.normalized_data = torch.from_numpy(self.normalized_data.reshape(1,1,-1))

                """ Predict the next price value """
                pred = self.model(self.normalized_data)
                out = pred.cpu().detach()

                if len(out.size()) > 1:
                    out = out.reshape(-1)

                """ Update the prices """
                predicted_price_inverse = scaler.inverse_transform(out.reshape(-1,1))
                self.predicted_prices.append(predicted_price_inverse[0][0])
                self.actual_prices.append(self.raw_data_queue[-1])

                """ Log out the predicted and last value """
                self.logger.debug(f"Predicted_Price: {predicted_price_inverse[0][0]}"
                                    f", Last Price:{self.raw_data_queue[-1]}")

                if predicted_price_inverse[0][0] > self.raw_data_queue[-1]:
                    self.logger.debug("Prediction: UP")
                else:
                    self.logger.debug("Prediction: DOWN")
                self.calculate_accuracy()

            elif self.model_type == 'keras':
                pred = self.model.predict(self.normalized_data.reshape(1,-1,1))
                if len(self.predicted_prices) == 0:
                    self.predicted_prices.append(0)
                self.predicted_prices.append(np.argmax(pred.reshape(-1)))

                if self.raw_data_queue[-1] > self.raw_data_queue[-2]:
                    self.actual_prices.append(0)
                else:
                    self.actual_prices.append(1)

                if np.argmax(pred.reshape(-1)) == 0:
                    self.logger.debug("Prediction: UP")
                else:
                    self.logger.debug("Prediction: DOWN")

                if len(self.actual_prices) >= 2:
                    acc = accuracy_score(y_true=self.actual_prices[1:], y_pred=self.predicted_prices[1:len(self.actual_prices)])
                    self.logger.debug(f"Accuracte keras predictions for"
                                        f" {len(self.actual_prices)} "
                                        f" till now: {acc}")

    def calculate_accuracy(self):
        if len(self.predicted_prices) > 1:
            y_pred = []
            y_true = []
            for i in range(0, len(self.predicted_prices)-1):
                if self.predicted_prices[i] > self.actual_prices[i]:
                    y_pred.append(0)
                else:
                    y_pred.append(1)

                if self.actual_prices[i+1] > self.actual_prices[i]:
                    y_true.append(0)
                else:
                    y_true.append(1)
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            self.logger.debug(f"Accuracte predictions for"
                                f" {len(self.predicted_prices)} "
                                f" till now: {accuracy}")

    def start(self, time_interval=60):
        """ Reinitialize the queue everytime this system is started """
        self.raw_data_queue = deque(maxlen=self.window_size)
        self.last_timestamp = None
        self.data_updated_flag = 0
    

        while True:
            if self.last_timestamp is None:
                """ Update the latest prices """ 
                self.logger.debug("\n")
                self.logger.debug("="*80)
                self.update_prices_data()
                self.logger.debug("="*80)
                self.logger.debug("\n")
            
            else:
                if (int(time.time()) - self.last_timestamp) >= time_interval:
                    self.logger.debug("\n")
                    self.logger.debug("="*80)
                    self.update_prices_data()
                    self.logger.debug("="*80)
                    self.logger.debug("\n")

                else:
                    self.logger.debug(
                        f"Time Since last Update: {time.time() - self.last_timestamp}")
                    time.sleep(1)




