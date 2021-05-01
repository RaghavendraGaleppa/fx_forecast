from collections import deque
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import torch
from tqdm import tqdm

def accuracy_stats(accuracy_dict):
    time_wise_accuracy = {}
    day_time_wise_accuracy = {}
    day_wise_accuracy = {}
    time_wise_accuracy = {}
    mean_time_wise_accuracy = {}
    for day in tqdm(accuracy_dict.keys()):
        day_time_wise_accuracy[day] = {}
        today_accuracies = []
        for time in accuracy_dict[day]:
            if time not in time_wise_accuracy:
                time_wise_accuracy[time] = []

            acc_score = accuracy_score(accuracy_dict[day][time]['real'],accuracy_dict[day][time]['pred'])
            
            day_time_wise_accuracy[day][time] = acc_score
            today_accuracies.append(acc_score)
            time_wise_accuracy[time].append(acc_score)

        day_wise_accuracy[day] = np.mean(today_accuracies)

    for t in time_wise_accuracy:
        mean_time_wise_accuracy[t] = np.mean(time_wise_accuracy[t])

    return {
                'day': day_wise_accuracy,
                'time': mean_time_wise_accuracy,
                'day_time': day_time_wise_accuracy
            }


def get_accuracy_score(inp, model, window_size, label_size):
    """
        inp should be a DataFrame
    """
    prices = deque(maxlen=window_size+1)
    accuracy_dict = {}

    for i in tqdm(range(len(inp))):
        prices.append(inp.start.iloc[i])

        if len(prices) == window_size+1:
            scaler = MinMaxScaler(feature_range=(0.1,0.9))
            ndarray = scaler.fit_transform(np.array([list(prices)[:-1]]).reshape(-1,1))
            print(ndarray)
            tensor = torch.tensor(ndarray.reshape(1,1,window_size))
            out = model(tensor)
            pred_price = out.cpu().detach()

            pred_rescale_value = scaler.inverse_transform(pred_price.reshape(1,1)).reshape(-1)

            _date = inp.date.iloc[i]
            _time = inp.time.iloc[i].split(':')[0]


            if not _date in accuract_dict:
                accuracy_dict[_date] = {}

            if not _time in accuracy_dict[_date][_time]:
                accuracy_dict[_date][_time] = {'real':[], 'pred':[]}


            if not type(pred_rescale_value) is int:
                pred_rescale_value = pred_rescale_value[0]
            
            if pred_rescale_value < prices[-2]:
                accuracy_dict[_date][_time]['pred'].append(0)
            else:
                accuracy_dict[_date][_time]['pred'].append(1)
                
            if prices[-1] < prices[-2]:
                accuracy_dict[_date][_time]['real'].append(0)
            else:
                accuracy_dict[_date][_time]['real'].append(1)

