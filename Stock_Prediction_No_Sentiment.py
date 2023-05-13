import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import yfinance as yf
from xgboost import XGBRegressor
from sklearn import metrics
import pickle

from Signal_Algorithm import RSI_Calc, getSignals


def getData(symbols):
    stock_data = []
    for ticker in symbols:
        data = yf.download(ticker, start='2010-01-01')
        stock_data.append(data)
    return stock_data


def cleanDataFrameandAddFeatures(stock_df):
    stock_df = stock_df.dropna()
    stock_df['HL_PCT'] = (stock_df['High'] - stock_df['Low']) / stock_df['Close'] * 100.0
    stock_df['PCT_change'] = (stock_df['Close'] - stock_df['Open']) / stock_df['Open'] * 100.0
    return stock_df


def splitData(stock_data, multiplier, predictors, targetCol):
    split_index = math.ceil(len(stock_data) * multiplier)
    train_df = stock_data.iloc[:split_index]
    test_df = stock_data.iloc[split_index:]
    X_train = train_df.loc[:, predictors].values
    X_test = test_df.loc[:, predictors].values
    y_train = train_df.loc[:, targetCol].values.reshape(-1, 1)
    y_test = test_df.loc[:, targetCol].values.reshape(-1, 1)
    y_train_dates = train_df.index.values
    y_test_dates = test_df.index.values

    return X_train, X_test, y_train, y_test, y_train_dates, y_test_dates


def scaleData(X_train, X_test):
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return X_train, X_test


def getPrediction(model, X_train, X_test, y_train , y_test):
    model.fit(X_train, y_train.ravel())
    predicted = model.predict(X_test)
    rootMeanSquared = np.sqrt(metrics.mean_squared_error(y_test, predicted))
    R_squared = model.score(X_test , y_test)
    return predicted, rootMeanSquared, R_squared

def getPredictionsWithSavedModel(modelName, X_test, y_test):
    model = pickle.load(open(f'../pickleFiles/Stock_{modelName}.pkl', 'rb'))
    predicted = model.predict(X_test)
    rootMeanSquared = np.sqrt(metrics.mean_squared_error(y_test, predicted))
    R_squared = metrics.r2_score(y_test, predicted)
    return predicted, rootMeanSquared, R_squared

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


# Testing the functions

params = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
params_svr = params_SVR = {"C": [0.001, 0.01, 0.1, 1, 10], "epsilon": [0.01, 0.1, 1, 10]}
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': GridSearchCV(Ridge(), param_grid=params),
    'SVR': GridSearchCV(SVR(kernel='linear'), param_grid=params_svr),
    'XGB': XGBRegressor(objective='reg:squarederror', n_estimators=1000)
}

tickers = ['AAPL', 'MSFT']


def runPredictionWithoutSentiment(modelName , getSignal , models=models):
    tickers = ['AAPL', 'MSFT']
    ml_model = models[modelName]
    stocks_data = getData(tickers)
    stock_clean = cleanDataFrameandAddFeatures(stocks_data[0])
    features = ['Open', 'High', 'Low']
    target_col = 'Adj Close'
    X_train, X_test, y_train, y_test, y_train_dates, y_test_dates = splitData(stock_clean, 0.8, features, target_col)
    X_train, X_test = scaleData(X_train, X_test)
    predictions, RMSE, R_Squared = getPrediction(ml_model, X_train, X_test, y_train, y_test)
    stocks , results, output_df = convertToPandas(stock_clean , y_test , predictions)
    print(output_df)
    print(mean_squared_error(y_test, predictions, squared=False))

    if getSignal:
        frames = RSI_Calc(output_df)
        buyingsignals, sellingdates = getSignals(frames)

        plt.figure(figsize=(12, 5))
        plt.title('Signals of ' + tickers[0] + ' using the model ' + modelName  )
        plt.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
        plt.plot(frames['Adj Close'], alpha=0.7)
        plt.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
        plt.plot(frames['Adj Close'], alpha=0.7)
        plt.show()
        profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
            buyingsignals].Open.values

        wins = [i for i in profits if i > 0]
        winning_rate = len(wins) / len(profits)
        return predictions, y_test, output_df, frames, buyingsignals, sellingdates, winning_rate
    return predictions, y_test ,  stocks, RMSE, R_Squared

def runPredictionWithoutSentimentWithSavedModels(modelName , getSignal , models=models):
    print("Loading model without sentiment...")
    tickers = ['AAPL', 'MSFT']
    stocks_data = getData(tickers)
    stock_clean = cleanDataFrameandAddFeatures(stocks_data[0])
    features = ['Open', 'High', 'Low']
    target_col = 'Adj Close'
    X_train, X_test, y_train, y_test, y_train_dates, y_test_dates = splitData(stock_clean, 0.8, features, target_col)
    X_train, X_test = scaleData(X_train, X_test)
    predictions, RMSE, R_Squared = getPredictionsWithSavedModel(modelName, X_test, y_test)
    stocks , results, output_df = convertToPandas(stock_clean , y_test , predictions)
    print(output_df)
    print(mean_squared_error(y_test, predictions, squared=False))

    if getSignal:
        frames = RSI_Calc(output_df)
        buyingsignals, sellingdates = getSignals(frames)

        plt.figure(figsize=(12, 5))
        plt.title('Signals of ' + tickers[0] + ' using the model ' + modelName  )
        plt.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
        plt.plot(frames['Adj Close'], alpha=0.7)
        plt.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
        plt.plot(frames['Adj Close'], alpha=0.7)
        plt.show()
        profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
            buyingsignals].Open.values

        wins = [i for i in profits if i > 0]
        winning_rate = len(wins) / len(profits)
        return predictions, y_test, output_df, frames, buyingsignals, sellingdates, winning_rate
    return predictions, y_test ,  stocks, RMSE, R_Squared
#print(runPredictionWithoutSentiment( 'SVR', False))
