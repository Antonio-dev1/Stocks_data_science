import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import yfinance as yf
from Signal_Algorithm import RSI_Calc,getSignals


def getData(symbols):
    stock_data = []
    for ticker in symbols:
        data = yf.download(ticker, start='2010-01-01')
        stock_data.append(data)
    return stock_data


def cleanDataFrame(stock_df):
    stock_df = stock_df.dropna()
    return stock_df


def splitData(stock_data, multiplier, predictors, targetCol):
    split_index = math.ceil(len(stock_data) * multiplier)
    train_df = stock_data.iloc[:split_index]
    test_df = stock_data.iloc[split_index:]
    X_train = train_df.loc[: , predictors].values
    X_test = test_df.loc[: , predictors].values
    y_train = train_df.loc[: , targetCol].values.reshape(-1, 1)
    y_test = test_df.loc[: , targetCol].values.reshape(-1, 1)
    y_train_dates = train_df.index.values
    y_test_dates = test_df.index.values

    return X_train, X_test, y_train, y_test, y_train_dates, y_test_dates


def scaleData(X_train, X_test):
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    return X_train, X_test


def getPrediction(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def convertPredictionsIntoPandas(predictions, test_dates, X_test, predictors):
    pred_df = pd.DataFrame(
        np.concatenate([X_test, predictions], axis=1), columns=predictors + ['Adj Close'], index=test_dates)
    return pred_df


def getPrecision():
    return 0


#Testing the functions

params = {"alpha": [0.001, 0.01, 0.1, 1, 10, 100]}
models = {
    'LinearRegression': LinearRegression(),
    'LogisticRegression': LogisticRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),

}


tickers = ['AAPL', 'MSFT']

# Get the Data
stocks_data = getData(tickers)
stock_clean = cleanDataFrame(stocks_data[0])
print(stock_clean.head())
features = ['Open', 'High' , 'Low']
target_col = 'Adj Close'
X_train , X_test , y_train , y_test , y_train_dates , y_test_dates = splitData(stock_clean , 0.8 , features , target_col)
X_train , X_test = scaleData(X_train , X_test)
output = getPrediction(models['LinearRegression'] , X_train , X_test , y_train)

output_df = convertPredictionsIntoPandas(output , y_test_dates , X_test , features)
print(output_df)
print(mean_squared_error(y_test , output , squared=False))
print("Test" , y_test)
print("Predicted" , output)
print(output_df)

#Get Signals from prediction part

frames = RSI_Calc(output_df)
buyingsignals, sellingdates = getSignals(frames)

plt.figure(figsize=(12, 5))
plt.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
plt.plot(frames['Adj Close'], alpha=0.7)
plt.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
plt.plot(frames['Adj Close'], alpha=0.7)
plt.show()

profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
    buyingsignals].Open.values


wins = [i for i in profits if i > 0]
winning_rate = len(wins) / len(profits)

print(winning_rate)