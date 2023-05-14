import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import accuracy_score, mean_squared_error
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from Signal_Algorithm import RSI_Calc, getSignals


def getData(start, tickers):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start)
    return data


def splitData(df):
    X = df.drop(['Adj Close', 'Close'], axis=1).values
    y = df['Adj Close'].values
    split = int(0.8 * len(df))
    train_data = df.iloc[:split]
    test_data = df.iloc[split:]

    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)

    target_col = 'Adj Close'
    n_steps = 60
    n_features = 5

    X_train = []
    y_train = []

    for i in range(n_steps, len(train_data_scaled)):
        X_train.append(train_data_scaled[i - n_steps:i, :n_features])
        y_train.append(train_data_scaled[i, df.columns.get_loc(target_col)])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = []
    for i in range(n_steps, len(test_data_scaled)):
        X_test.append(test_data_scaled[i - n_steps:i, :n_features])
        y_test.append(test_data_scaled[i, df.columns.get_loc(target_col)])
    X_test, y_test = np.array(X_test), np.array(y_test)

    return X_train, X_test, y_train, y_test


def getPredictions(n_steps, n_features, X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=5, batch_size=32)
    y_pred = model.predict(X_test).flatten()
    print(y_pred.shape)
    # rescale predictions and actual values
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('Test RMSE: %.3f' % rmse)

    print(y_pred)
    print(y_test)
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.legend(['Predictions', 'Actual'])
    plt.show()

    return y_pred,rmse

def getPredictionsWithSavedModel(n_steps, n_features, X_train, X_test, y_train, y_test):
    model = pickle.load(open('../pickleFiles_crypto/Crypto_LSTM.pkl', 'rb'))
    y_pred = model.predict(X_test).flatten()
    print(y_pred.shape)
    # rescale predictions and actual values
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('Test RMSE: %.3f' % rmse)

    print(y_pred)
    print(y_test)
    plt.plot(y_pred)
    plt.plot(y_test)
    plt.legend(['Predictions', 'Actual'])
    plt.show()

    return y_pred,rmse


def convertToPickle(model):
    pickle.dump(model, open('./pickleFiles/Stock_LSTM.pkl', 'wb'))

def convertToPandas(df, real_prices, predicted_prices):
    stocks = pd.DataFrame({
        "Real": real_prices.ravel(),
        "Adj Close": predicted_prices.ravel()
    }, index=df.index[-len(real_prices):])

    results = pd.DataFrame({
        "Adj Close": predicted_prices.ravel()
    }, index=df.index[-len(real_prices):])
    originaldata = df.loc[stocks.index, ['Close', 'Open', 'High', 'Low']]
    output_df = pd.concat([originaldata, results], axis=1)
    return stocks, results, output_df


allTickers = ['AAPL']
oneTicker = 'AAPL'


def runLSTM(getSignal, tickers=allTickers, stockInChoice=oneTicker):
    start = dt.datetime(2010, 1, 1)
    stock_dict = getData(start, tickers)
    stock_df = stock_dict[stockInChoice]
    X_train, X_test, y_train, y_test = splitData(stock_df)
    y_pred,RMSE = getPredictions(60, 5, X_train, X_test, y_train, y_test)
    stocks,results,output_df = convertToPandas(stock_df,y_test,y_pred)

    if getSignal:
        stocks, results, output_df = convertToPandas(stock_df, y_test, y_pred)
        frames = RSI_Calc(output_df)
        buyingsignals, sellingdates = getSignals(frames)
        profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
            buyingsignals].Open.values
        wins = [i for i in profits if i > 0]
        winning_rate = len(wins) / len(profits)
        plt.figure(figsize=(12, 5))
        plt.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
        plt.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
        plt.plot(frames['Adj Close'], alpha=0.7)
        return y_pred, y_test, output_df, frames, buyingsignals, sellingdates, winning_rate
    return y_pred, y_test ,  stocks, RMSE

def runLSTMWithSavedModel(getSignal, tickers=allTickers, stockInChoice=oneTicker):
    start = dt.datetime(2010, 1, 1)
    stock_dict = getData(start, tickers)
    stock_df = stock_dict[stockInChoice]
    X_train, X_test, y_train, y_test = splitData(stock_df)
    y_pred,RMSE = getPredictionsWithSavedModel(60, 5, X_train, X_test, y_train, y_test)
    stocks,results,output_df = convertToPandas(stock_df,y_test,y_pred)

    if getSignal:
        stocks, results, output_df = convertToPandas(stock_df, y_test, y_pred)
        frames = RSI_Calc(output_df)
        buyingsignals, sellingdates = getSignals(frames)
        profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
            buyingsignals].Open.values
        wins = [i for i in profits if i > 0]
        winning_rate = len(wins) / len(profits)
        plt.figure(figsize=(12, 5))
        plt.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
        plt.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
        plt.plot(frames['Adj Close'], alpha=0.7)
        return y_pred, y_test, output_df, frames, buyingsignals, sellingdates, winning_rate
    return y_pred, y_test ,  stocks, RMSE

#runLSTM(True)
