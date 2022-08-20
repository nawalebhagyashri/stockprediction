"""
LSTM RNN model

created by: Ban Luong
"""
import tensorflow as tf
from tensorflow import keras

import yfinance as yf
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import datetime

mpl.rcParams['figure.figsize'] = (16, 10)
mpl.rcParams['axes.grid'] = False

import plotly.offline as py
# That's line needed if you use jupyter notebook (.ipynb):
py.init_notebook_mode(connected=True)

import plotly.graph_objects as go
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tickerSymbol = 'NFLX'
today = datetime.date.today()
start = datetime.datetime(today.year-5,today.month,today.day)


def candleStick(symbol, startdate, enddate):
    tickerData = yf.Ticker(symbol)
    df = tickerData.history(period='1d', start=startdate, end=enddate)
    
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])

    fig.update_layout(
        title= {
            'text': symbol,
          'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
          font=dict(
            family="Times New Roman",
            size=20,
            color="#7f7f7f"
            )
        )

    fig.show()
def dataFrame(symbol, period, startdate, enddate):
    
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    tickerData = yf.Ticker(symbol)
    tickerDf = tickerData.history(period=period, start=startdate, end=enddate)
    
    return tickerDf[columns]

df = dataFrame(tickerSymbol, '1d', start, today)

def plotChart(df):
    plt.figure(figsize=(15,6))
    df['Close'].plot()
    plt.title('Stock Price over Time (%s)'%tickerSymbol, fontsize=20)
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Price', fontsize=16)
    
    for year in range(2015,2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)

plotChart(df)

df1 = df[['Close']]

train_size = int(len(df1) * 0.8)
test_size = len(df1) - train_size

# Standarize dataset values to reduce loss
train_mean = df1[:train_size].mean()
train_std = df1[:train_size].std()

df1 = (df1-train_mean)/train_std

train, test = df1.iloc[0:train_size], df1.iloc[train_size:len(df1)]
print(len(train), len(test))

train

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 1

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.Close, time_steps)
X_test, y_test = create_dataset(test, test.Close, time_steps)

print(X_train.shape, y_train.shape)

df = dataFrame(tickerSymbol, '1d', start, today)
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def build_model(LSTM_unit, dropout, lr, train):
    model = Sequential()
    model.add(LSTM(units=LSTM_unit, return_sequences=True, input_shape=(train.shape[-2:])))
    model.add(Dropout(dropout))

    model.add(LSTM(units=LSTM_unit, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(units=LSTM_unit, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(LSTM(units=LSTM_unit))
    model.add(Dropout(dropout))

    model.add(Dense(units=1))

    model.compile(optimizer=Adam(lr), loss='mean_squared_error')

    return model


model = build_model(50, 0.2, 0.001, X_train)

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    verbose=1,
    shuffle=False
)
