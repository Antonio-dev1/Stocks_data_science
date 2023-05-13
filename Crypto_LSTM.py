import pickle

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

import yfinance as yf


def getData(ticker):
    data = yf.download(ticker, start='2009-01-01')
    return data


def normalise_zero_base(continuous):
    return continuous / continuous.iloc[0] - 1


def normalise_min_max(continuous, data):
    return (continuous - continuous.min()) / (data.max() - continuous.min())


def extract_window_data(continuous, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(continuous) - window_len):
        tmp = continuous[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(data, aim, window_len=10, zero_base=True, test_size=0.2):
    train_data = data.iloc[:200]
    test_data = data.iloc[200:]
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[aim][window_len:].values
    y_test = test_data[aim][window_len:].values
    if zero_base:
        y_train = y_train / train_data[aim][:-window_len].values - 1
        y_test = y_test / test_data[aim][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('BTC/USD', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16);
    plt.show()


def convertToPickle(model):
    pickle.dump(model, open('./pickleFiles/Crypto_LSTM.pkl', 'wb'))


def generate_signals(y_test, preds):
    price_diff = preds - y_test
    buy_signals = np.zeros_like(price_diff, dtype=bool)
    sell_signals = np.zeros_like(price_diff, dtype=bool)
    buy_signals[::100] = price_diff[::100] < 0
    sell_signals[::100] = price_diff[::100] > 0
    return buy_signals, sell_signals


def calculate_next_day_price(model, X_test):
    next_day_pred = model.predict(X_test[-1].reshape(1, -1))
    return next_day_pred[0]


def runCryptoLSTM(getSignal):
    data = getData('BTC-USD')
    aim = 'Close'
    window_len = 5
    test_size = 0.2
    zero_base = True
    lstm_neurons = 50
    epochs = 20
    batch_size = 32
    loss = 'mse'
    dropout = 0.24
    optimizer = 'adam'
    train_data, test_data, X_train, X_test, y_train, y_test = prepare_data(data, aim, window_len=window_len,
                                                                           zero_base=zero_base, test_size=test_size)

    model = build_lstm_model(
        X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
        optimizer=optimizer)
    modelfit = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1,
        shuffle=True)

    targets = test_data[aim][window_len:]
    preds = model.predict(X_test).squeeze()
    mse = mean_absolute_error(preds, y_test)

    preds = test_data[aim].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual', 'prediction', lw=2)
    if getSignal:
        if getSignal:
            buy_signals, sell_signals = generate_signals(y_test, preds)
            return preds, test_data, window_len, buy_signals, sell_signals

    return preds, targets, mse

# runCryptoLSTM(False)
