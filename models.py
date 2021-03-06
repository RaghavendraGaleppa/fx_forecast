import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from sklearn.metrics import accuracy_score
import numpy as np

def load_model_from_checkpoint(checkpoint_path):
    model = None
    model_config = checkpoint_path.split('.')[0].split('/')[-1].split('_')
    model_name = model_config[0]
    window_size = int(model_config[1])
    label_size = int(model_config[2])
    
    if checkpoint_path.split('.')[1] == 'pth':
        model_type = 'keras'
        if model_name == 'CNNLSTMModel':
            model = cnn_lstm_model((window_size,1), label_size)
            model.load_weights(checkpoint_path)

        if model_name == "LSTMModel":
            model = lstm_model((window_size, 1))
            model.load_weights(checkpoint_path)

        if model_name == "TCNModel":
            model = tcn_model((window_size,1), label_size)
            model.load_weights(checkpoint_path)

        if model_name == 'LinearModel':
            model = linear_model((window_size,1))
            model.load_weights(checkpoint_path)
    else:
        model_type = 'torch'

        if model_name == 'BaseLinearModel':
            model = BasicLinearModel.load_from_checkpoint(
                checkpoint_path,
                window_size=window_size,
                label_size=label_size
            ).double()

    return model, window_size, label_size, model_type
        
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
        loss_scores = [l['loss'] for l in outputs]
        accuracy_scores = [a['accuracy_score'] for a in outputs]

        loss = torch.stack(loss_scores).mean()
        accuracy = np.mean(accuracy_scores)

        self.trainer.progress_bar_callback.main_progress_bar.write(
            f"Epoch {self.trainer.current_epoch} "
            f"validation_loss={loss.item()} "
            f"validation_accuracy={accuracy}"
        )

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
        pred = []
        real = []
        for i in range(len(x)):
            print(x[i])
            print(logits[i], labels[i])
            if x.squeeze()[i,-1].item() < logits[i][0].item():
                pred.append(0)
            else:
                pred.append(1)

            if x.squeeze()[i,-1].item() < labels[i].item():
                real.append(0)
            else:
                real.append(1)
        
        return accuracy_score(y_true=real, y_pred=pred)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.mse_loss(logits, y)
        #accuracy = self.accuracy_metric(x, logits.cpu().detach(), y)
        accuracy = 0
        self.log('train_loss', loss)
        self.log('accuracy_score', accuracy)
        return {'loss':loss, 'accuracy_score':accuracy}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.mse_loss(logits, y)
        accuracy = self.accuracy_metric(x, logits.cpu().detach(), y)
        self.log('val_loss', loss)
        self.log('accuracy_score', accuracy)
        return {'loss':loss, 'accuracy_score':accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


from keras.models import Model,Sequential
from keras import optimizers
from keras.utils import to_categorical
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,Conv1D,BatchNormalization,MaxPooling1D,LSTM,Dense,Activation,Layer, Dropout, Flatten
import tensorflow as tf
from tcn import TCN, tcn_full_summary

def tcn_model(input_shape, label_size=1):
    model = Sequential([
    TCN(input_shape=input_shape,
        kernel_size=2,
        use_skip_connections=False,
        use_batch_norm=False,
        use_weight_norm=False,
        use_layer_norm=False
        ),
    Dense(label_size, activation='linear')
    ])

    return model

def cnn_lstm_model(input_shape, num_classes=2):
    model = Sequential()

    model.add(Conv1D(filters = 64,kernel_size = 3,strides=1,padding='same', input_shape=input_shape))	
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dropout(0.15))

    #LFLB2
    model.add(Conv1D(filters=64, kernel_size = 3, strides=1,padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling1D(pool_size = 2, strides = 2))

    model.add(LSTM(units=128, activation='relu', return_sequences=True)) 

    model.add(Dropout(0.15))

    model.add(LSTM(units=128, activation='relu')) 

    if num_classes > 1:
        model.add(Dense(units=num_classes,activation='softmax'))
    else:
        model.add(Dense(units=num_classes))

    #Model compilation	
    return model

def lstm_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=input_shape))
    model.add(tf.keras.layers.Dense(1))
    return  model

def residual_cnn(input_shape):
    model = tf.keras.models.Sequential()

def linear_model(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1))
    return model
