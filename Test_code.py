from gc import callbacks
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential

def LSTM_Model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(x_tr.shape[1], x_tr.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(LSTM(32, return_sequences=True))
    model.add(Dropout(0.2))  # adding dropout layer to reduce overfitting
    model.add(BatchNormalization())

    model.add(Dense(y_tr.shape[1]))  # output layer with 1 output.

    opt = tf.keras.optimizers.Adam(learning_rate=0.005, decay=1e-6)

    model.compile(loss='mean_squared_error',
                  optimizer=opt,
                  metrics=['mae'])
    return model



if __name__ == "__main__":
    n_future = 1
    n_past = 60

    data = pd.read_csv("Kucoin_BTCUSDT_minute.csv", header=1)
    print(data.shape)
    print('Number of null values: ', data.isnull().sum().sum())  # checking if there are any null values

    times = sorted(data.index.values)
    last_5pct = times[-int(0.05 * len(times))]

    validation_data_1 = data[(data.index) >= last_5pct]
    training_data_1 = data[(data.index) < last_5pct]

    print("times", last_5pct)
    print(validation_data_1.shape)
    print(training_data_1.shape)


    training_data = np.array(training_data_1)
    print("training data", training_data)
    print(training_data.shape)


    scaler = StandardScaler()
    scaler = scaler.fit(training_data)
    scaled_data = scaler.transform(training_data)

    sc_scaler = StandardScaler()
    sc_scaler.fit_transform(training_data[:, 0:1])

    print("scaled_data", scaled_data)
    print("sc_scaled", sc_scaler)

    validation_data = np.array(validation_data_1)
    print(validation_data.shape)

    val_scaled_data = scaler.transform(validation_data)

    x_tr = []
    y_tr = []

    for i in range(n_past, len(scaled_data) - n_future + 1):
        x_tr.append(scaled_data[i - n_past:i, 0: scaled_data.shape[1]])
        y_tr.append(scaled_data[i + n_future - 1:i + n_future, 0])

    x_tr, y_tr = np.array(x_tr), np.array(y_tr)

    x_val = []
    y_val = []

    for i in range(n_past, len(val_scaled_data) - n_future + 1):
        x_val.append(val_scaled_data[i - n_past:i, 0: validation_data.shape[1]])
        y_val.append(val_scaled_data[i + n_future - 1:i + n_future, 0])

    x_val, y_val = np.array(x_val), np.array(y_val)

    model = LSTM_Model()

    history = model.fit(x_tr, y_tr, epochs=1, batch_size=64, validation_data = (x_val, y_val))
    prediction = model.predict(x_val).reshape(-1, 1)

    y_pred_future = sc_scaler.inverse_transform(prediction)
    actaul_val = sc_scaler.inverse_transform(y_val)
    predicted_values = list(y_pred_future)
    print("predicted values", predicted_values)

    actual_values = list(actaul_val)
    print("actual_values", actual_values)







