import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

device_name = tf.test.gpu_device_name()
print(device_name)
if device_name != '/device:GPU:0':
    print('no gpu found')

else:
    print("GPU IS PRESENT")


#reading dataset:

data = pd.read_csv("Kucoin_BTCUSDT_minute.csv", header=1)
print(data.shape)

data['date'] = pd.to_datetime(data['date']) # conveting date column to datetime format
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


#Normalizing the values of all OHLC values between 0 and 1

scaler = StandardScaler()
scaler = scaler.fit(data)
data_training = scaler.transform(data)

#splitting data into training

x_tr = []
y_tr = []

n_future = 1 # number of minutes we want to predict in future
n_past = 60 # number of minutes we want to use to predict future

for i in range(n_past, len(data_training) - n_future + 1):
    x_tr.append(data_training[i-n_past:i, 0: data_training.shape[1]])
    y_tr.append(data_training[i+n_future-1:i+n_future, 0])


x_tr, y_tr = np.array(x_tr), np.array(y_tr) # converting to array

print('x_tr shape == {}.'.format(x_tr.shape))
print('y_tr shape == {}.'.format(y_tr.shape))

def model1():
    with tf.device('/device:GPU:0'):
        model = Sequential()
        model.add(LSTM(16, activation='relu', input_shape=(x_tr.shape[1], x_tr.shape[2]), return_sequences=True)) #input LSTM layer with 16 neurons, relu as activation function and input shape is the shape of our data
        model.add(LSTM(8, activation='relu', return_sequences=True)) # hidden LSTM layer, with 8 neurons and relu as activation function.
        model.add(Dropout(0.3)) # adding dropout layer to reduce overfitting
        model.add(Dense(y_tr.shape[1])) # output layer with 1 output.

        model.compile(optimizer='adam', loss='mse')

        history = model.fit(x_tr, y_tr, epochs=1, batch_size=64, validation_split=0.1, verbose=1)

model1()






