'''this program is done based on regression model'''


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

device_name = tf.test.gpu_device_name()
print(device_name)
if device_name != '/device:GPU:0':
    print('no gpu found')

else:
    print("GPU IS PRESENT")


n_future = 1
n_past = 60
def splitting_data(df):
    scaler = StandardScaler()
    scaler = scaler.fit(df)
    scaled_data = scaler.transform(df)

    x = []
    y = []

    for i in range(n_past, len(scaled_data) - n_future + 1):
        x.append(scaled_data[i - n_past:i, 0: scaled_data.shape[1]])
        y.append(scaled_data[i + n_future - 1:i + n_future, 0])

    return np.array(x), np.array(y)


#reading dataset:

data = pd.read_csv("Kucoin_BTCUSDT_minute.csv", header=1)
print(data.shape)

data['date'] = pd.to_datetime(data['date'])# conveting date column to datetime format
print('Number of null values: ', data.isnull().sum().sum()) #checking if there are any null values

data = data.sort_values(by=["date"], ascending=True) #sorting the data according to date in ascending order to make it easy to devide data for training and testing.

data.drop(['unix','symbol', 'Volume BTC', 'Volume USDT'], axis=1, inplace=True)

data.index = data.pop('date') #making date column as an index value
#print(data.head())

#plotting Closing price history

"""plt.title('close price history')
plt.xlabel('date')
plt.ylabel('Close price')
plt.xticks(rotation=45)
plt.plot(data.index, data['close'])
plt.show()
"""

times = sorted(data.index.values)
last_5pct = times[-int(0.05*len(times))]

validation_data = data[(data.index)>=last_5pct]
training_data = data[(data.index)<last_5pct]

x_tr, y_tr = splitting_data(training_data)
x_val, y_val = splitting_data(validation_data)

x_tr = np.asarray(x_tr)
y_tr = np.asarray(y_tr)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)


'''scaler = StandardScaler()
scaler = scaler.fit(data)
data_training = scaler.transform(data)

x_tr = []
y_tr = []

n_future = 1 # number of minutes we want to predict in future
n_past = 60 # number of minutes we want to use to predict future

for i in range(n_past, len(data_training) - n_future + 1):
    x_tr.append(data_training[i-n_past:i, 0: data_training.shape[1]])
    y_tr.append(data_training[i+n_future-1:i+n_future, 0])


x_tr, y_tr = np.array(x_tr), np.array(y_tr) # converting to array

print('x_tr shape == {}.'.format(x_tr.shape))
print('y_tr shape == {}.'.format(y_tr.shape))'''

def model1():
    with tf.device('/device:GPU:0'):

        model = Sequential()
        model.add(LSTM(64, input_shape=x_tr.shape[1:], return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(LSTM(32, return_sequences=True))
        model.add(Dropout(0.2))# adding dropout layer to reduce overfitting
        model.add(BatchNormalization())

        model.add(Dense(y_tr.shape[1]))# output layer with 1 output.

        opt = tf.keras.optimizers.Adam(learning_rate=0.005, decay=1e-6)

        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['mae'])

        history = model.fit(x_tr, y_tr, epochs=5, batch_size=64, validation_data=(x_val, y_val))


        #plotting training and validation ccuracy and loss at each epoch
        loss = history.history['loss']
        validation_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'y', label='training loss')
        plt.plot(epochs, validation_loss, 'r', label='validation loss')
        plt.title('TRaining and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = history.history['mae']
        val_acc = history.history['val_mae']
        plt.plot(epochs, acc, 'y', label='Training Mae')
        plt.plot(epochs, val_acc, 'r', label='Val_Mae')
        plt.title('TRaining and validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('ACCURACY')
        plt.legend()
        plt.show()

model1()






