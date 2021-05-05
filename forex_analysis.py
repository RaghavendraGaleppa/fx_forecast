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
    def __init__(self, model_checkpoint_path, simulation=None, categorical=False):


        self.model, self.window_size, self.label_size, self.model_type = load_model_from_checkpoint(
            checkpoint_path=model_checkpoint_path,
        )

        self.simulation = simulation
        self.categorical = categorical

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
        self.prediction_labels = ['UP', 'DOWN']


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
            """
            Flow:
                - Normalize the data
                - Next price prediction: ouputs a numpy array of predictions
                    - If model is pytorch:
                        - convert the ndarray into a tensor and into appropriate shape
                        - get the prediction
                        - convert the prediction back to an (n_batch, n_labels) np array
                    - If the model is keras
                        - Pass the normalized data into the model and get the prediction
                - Based on labels type:
                    - If the labels are categorical:
                        - Append the argmax of the prediction into predicted prices array
                        - Append the argmax of last two values in raw_data_queue into actual values array
                    - If the label is regression
                        - Reverse transform it into the original price
                        - Based on whether the value is greater than latest price, append its labels into
                        the predicted prices array
                        - Append the argmax of last two values in raw_data_queue into actual values array

                - Log out the predictions
                - Log out the accuracy till now
            """

            """ Normalize the prices for next price prediction """
            data_ndarray = np.array(self.raw_data_queue)

            """ Create a scaler and fit the raw data """
            scaler = MinMaxScaler(feature_range=(0.1,0.9))
            self.normalized_data = scaler.fit_transform(data_ndarray.reshape(-1,1)) 
            self.predictions_made = None

            """ Based on the models, get the predictions """
            if self.model_type == 'torch':
                self.normalized_data = torch.from_numpy(self.normalized_data.reshape(1,1,-1))

                """ Predict the next price value """
                pred = self.model(self.normalized_data)
                self.predictions_made = pred.cpu().detach().numpy()

            elif self.model_type == 'keras':
                self.predictions_made = self.model.predict(self.normalized_data.reshape(1,-1,1))

            """ Handle categorical and regressional predictions """
            if self.categorical is True:
                pred_label = np.argmax(self.predictions_made.reshape(-1))
                self.logger.debug(f"Predictions made: {self.predictions_made.reshape(-1)}")

            else:
                raw_predicted_price = scaler.inverse_transform(self.predictions_made.reshape(-1,1))
                pred_label = np.argmax([raw_predicted_price[0,0], self.raw_data_queue[-1]])
                self.logger.debug(f"Predicted Price: {raw_predicted_price[0,0]}, Last Price: {self.raw_data_queue[-1]}")

            actual_label = np.argmax(list(self.raw_data_queue)[::-1][:2])

            if len(self.predicted_prices) == 0:
                self.predicted_prices.append(0)
            self.predicted_prices.append(pred_label)
            self.actual_prices.append(actual_label)

            self.logger.debug(f"Raw Prices: {self.raw_data_queue}")
            self.logger.debug(f"Normalized Prices: {self.normalized_data.reshape(-1)}")
            self.logger.debug(f"Movement: {self.prediction_labels[pred_label]}")

            if len(self.actual_prices) >= 2:
                acc = accuracy_score(y_true=self.actual_prices[1:], y_pred=self.predicted_prices[1:len(self.actual_prices)])
                self.logger.debug(f"Accuracte keras predictions for"
                                    f" {len(self.actual_prices)} "
                                    f" till now: {acc}")


    def start(self, time_interval=60, fill_raw_data_queue=True):
        """ Reinitialize the queue everytime this system is started """
        self.raw_data_queue = deque(maxlen=self.window_size)
        if fill_raw_data_queue is True:
            for i in range(self.window_size):
                self.raw_data_queue.append(0)
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




