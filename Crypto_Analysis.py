import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
import pickle

def download_data(tickers):
    data = yf.download(tickers , start='2010-01-01')
    return data


def extract_features(data):
    data = data.reset_index()
    data = data.set_index('Date')
    data.index = pd.to_datetime(data.index, unit='ns')
    # data = data.loc[:, ['Close']]
    data['High-Low'] = data['High'] - data['Low']
    data['Close-Open'] = data['Close'] - data['Open']
    return data


def split_data(data, train_size):
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    return train_data, test_data


def prepare_data(train_data, test_data, window_size):
    X_train = np.array([
        train_data[i:i + window_size][['Close', 'High-Low', 'Close-Open']].values.ravel()
        for i in range(len(train_data) - window_size)
    ])
    y_train = train_data['Close'].values[window_size:]
    X_test = np.array([
        test_data[i:i + window_size][['Close', 'High-Low', 'Close-Open']].values.ravel()
        for i in range(len(test_data) - window_size)
    ])
    y_test = test_data['Close'].values[window_size:]
    return X_train, y_train, X_test, y_test



def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    preds = model.predict(X_test)
    return preds


def evaluate(y_test, preds):
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mae, mse, r2


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



def convertToPickle(model):
    pickle.dump(model, open('pickleFiles_crypto/Crypto_XGB.pkl', 'wb'))



# def plot_results(data, window_size, train_size, y_test, preds):
#     plt.plot(data.index[window_size + train_size:], data['Close'][window_size + train_size:], label='Actual',
#              linewidth=2)
#     plt.plot(test_data.index[window_size:], preds, label='Predicted', linewidth=2)
#     plt.xlabel('Date')
#     plt.ylabel('BTC/USD')
#     plt.title('BTC/USD Prediction using Linear Regression with Rolling Window')
#     plt.legend()
#     plt.show
#
#
# def plot_signals(data, buy_signals, sell_signals):
#     fig, ax = plt.subplots()
#     ax.plot(test_data.index[window_size:], preds, label='Predicted', linewidth=2)
#     ax.scatter(test_data.index[window_size:][buy_signals], preds[buy_signals], color='green', label='Buy Signal',
#                marker='^', zorder=3)
#     ax.scatter(test_data.index[window_size:][sell_signals], preds[sell_signals], color='red', label='Sell Signal',
#                marker='v', zorder=3)
#     ax.legend()
#     plt.show()
#
#
# # Download the data
# tickers = ['BTC-USD']
# data = download_data(tickers)
#
# # Extract relevant features
# data = extract_features(data)
#
# # Split the data into training and test sets
# train_data, test_data = split_data(data, 200)
#
# # Prepare the data for linear regression with a rolling window
# window_size = 5
# X_train, y_train, X_test, y_test = prepare_data(train_data, test_data, window_size)
#
# # Create and train the linear regression model
# model = train_model(X_train, y_train)
#
# # Make predictions on the test set
# preds = predict(model, X_test)
#
# # Compute evaluation metrics
# mae, mse, r2 = evaluate(y_test, preds)
# print("Mean Absolute Error (MAE):", mae)
# print("Mean Squared Error (MSE):", mse)
# print("R-squared Score (R2):", r2)
#
# # Generate buy and sell signals
# buy_signals, sell_signals = generate_signals(y_test, preds)
#
# # Plot the actual and predicted values in a separate window
# plot_results(data, window_size, len(train_data), y_test, preds)
# plt.show()
#
# # Create a new figure for the signals
# plot_signals(data, buy_signals, sell_signals)
# plt.show()

lasso_params = {"alpha": [0.01, 0.1, 1, 10, 100, 1000]}
params_SVR = {"C": [0.001, 0.01, 0.1, 1, 10], "epsilon": [0.01, 0.1, 1, 10]}

models = {'LinearRegression': LinearRegression(), 'Lasso': GridSearchCV(Lasso(), param_grid=lasso_params),
          'Ridge': GridSearchCV(Ridge(), param_grid=lasso_params), 'SVR': GridSearchCV(SVR(
        kernel='linear'), param_grid=params_SVR), 'XGB': XGBRegressor(objective='reg:squarederror', n_estimators=1000),
          'RandomForest': RandomForestRegressor(n_estimators=200, min_samples_split=50, random_state=1)}


def runCryptoAnalysis(getSignal, modelName, allModels=models):
    tickers = 'BTC-USD'
    data = download_data(tickers)
    ml_model = allModels[modelName]

    # Extract relevant features
    data = extract_features(data)

    # Split the data into training and test sets
    train_data, test_data = split_data(data, 200)

    # Prepare the data for linear regression with a rolling window
    window_size = 5
    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data, window_size)

    # Create and train the linear regression model
    model = train_model(X_train, y_train , ml_model)

    # Make predictions on the test set
    preds = predict(model, X_test)


    # Compute evaluation metrics
    mae, mse, r2 = evaluate(y_test, preds)

    if getSignal:
        buy_signals, sell_signals = generate_signals(y_test, preds)
        return preds, test_data, window_size, buy_signals, sell_signals
    else:
        next_day_price = calculate_next_day_price(model, X_test)
        return preds, test_data, train_data, y_test, window_size, mae, mse, r2, data, X_test, next_day_price

def runCryptoAnalysisWithSavedModels(getSignal, modelName, allModels=models):
    print("Loading saved crypto models...")
    tickers = 'BTC-USD'
    data = download_data(tickers)
    ml_model = allModels[modelName]

    # Extract relevant features
    data = extract_features(data)

    # Split the data into training and test sets
    train_data, test_data = split_data(data, 200)

    # Prepare the data for linear regression with a rolling window
    window_size = 5
    X_train, y_train, X_test, y_test = prepare_data(train_data, test_data, window_size)

    # Create and train the linear regression model
    model = pickle.load(open(f'../pickleFiles_crypto/Crypto_{modelName}.pkl', 'rb'))

    # Make predictions on the test set
    preds = predict(model, X_test)

    # Compute evaluation metrics
    mae, mse, r2 = evaluate(y_test, preds)

    if getSignal:
        buy_signals, sell_signals = generate_signals(y_test, preds)

        return preds,test_data, window_size,buy_signals, sell_signals
    else:
        next_day_price = calculate_next_day_price(model, X_test)
        return preds, test_data, train_data, y_test, window_size, mae, mse, r2, data, X_test, next_day_price



# preds, test_data, train_data, y_test, window_size, mae, mse, r2, data, X_test = runCryptoAnalysis(False , 'XGB')
