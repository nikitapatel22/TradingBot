'''This program is done based on classification algorith'''

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import random

import warnings
from collections import deque
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

warnings.filterwarnings("ignore")
n_past = 60
n_future = 3

def target_classify(current, future):
    if future > current:
        return 1
    else:
        return 0

def pre_processing(df):
    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    seq_data = []
    prev_days = deque(maxlen=n_past)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == n_past:
            seq_data.append([np.array(prev_days), i[-1]])

    random.shuffle(seq_data)
    x = []
    y = []

    for seq, target in seq_data:
        x.append(seq)
        y.append(target)

    return np.array(x), y


data = pd.read_csv("Kucoin_BTCUSDT_minute.csv", header=1)
print(data.shape)

data['date'] = pd.to_datetime(data['date']) # conveting date column to datetime format
print('Number of null values: ', data.isnull().sum().sum()) #checking if there are any null values

#data = data.sort_values(by=["date"], ascending=True) #sorting the data according to date in ascending order to make it easy to devide data for training and testing.

data.drop(['unix','symbol', 'Volume BTC', 'Volume USDT'], axis=1, inplace=True)

data.index = data.pop('date')

#print(data.head())

data['future'] = data['close'].shift(-n_future)

data['target'] = list(map(target_classify, data['close'], data['future']))



print(data.head())

data.drop(['future'], axis=1, inplace=True)
#print(data.head())

times = sorted(data.index.values)
last_pct = times[-int(0.1*len(times))]
#print(last_5pct)

validation_data = data[(data.index)>=last_pct]
training_data = data[(data.index)<last_pct]

train_x, train_y = pre_processing(training_data)
val_x, val_y = pre_processing(validation_data)

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
val_x = np.asarray(val_x)
val_y = np.asarray(val_y)

#print(train_y)
print(len(train_x), len(val_x))
def LSTM_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=train_x.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.005, decay=1e-6)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, batch_size=32, epochs=3, validation_data=(val_x, val_y))
    print(history)

    #score evaluation

    score = model.evaluate(val_x, val_y, verbose=0)
    print('Validation loss:', score[0])
    print('Validation accuracy:', score[1])


LSTM_model()