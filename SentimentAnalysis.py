import pickle

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np
from Signal_Algorithm import RSI_Calc, getSignals
from sklearn.svm import SVR


def window_data(df, window, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number):
    # Create empty lists "X_close", "X_polarity", "X_volume" and y
    X_close = []
    X_polarity = []
    X_volume = []
    y = []
    for i in range(len(df) - window):
        # Get close, ts_polarity, tw_vol, and target in the loop
        close = df.iloc[i:(i + window), feature_col_number1]
        ts_polarity = df.iloc[i:(i + window), feature_col_number2]
        tw_vol = df.iloc[i:(i + window), feature_col_number3]
        target = df.iloc[(i + window), target_col_number]

        # Append values in the lists
        X_close.append(close)
        X_polarity.append(ts_polarity)
        X_volume.append(tw_vol)
        y.append(target)

    return np.hstack((X_close, X_polarity, X_volume)), np.array(y).reshape(-1, 1)


def convertPredictionsIntoPandas(predictions, test_dates, X_test, predictors):
    pred_df = pd.DataFrame(
        np.concatenate([X_test, predictions], axis=1), columns=predictors + ['Adj Close'], index=test_dates)
    return pred_df


def getPredictions(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train.ravel())
    predicted = model.predict(X_test)
    rootMeanSquared = np.sqrt(metrics.mean_squared_error(y_test, predicted))
    R_squared = metrics.r2_score(y_test, predicted)
    return predicted, rootMeanSquared, R_squared

def getPredictionsWithSavedModel(modelName, X_test, y_test):
    model = pickle.load(open(f'../pickleFiles_sentiment/Stock_{modelName}.pkl', 'rb'))
    predicted = model.predict(X_test)
    rootMeanSquared = np.sqrt(metrics.mean_squared_error(y_test, predicted))
    R_squared = metrics.r2_score(y_test, predicted)
    return predicted, rootMeanSquared, R_squared

def splitData(X, y):
    X_split = int(0.2 * len(X))
    y_split = int(0.2 * len(y))

    # Set X_train, X_test, y_train, t_test
    X_train = X[: X_split]
    X_test = X[X_split:]
    y_train = y[: y_split]
    y_test = y[y_split:]
    x_train_scaler = MinMaxScaler()
    x_test_scaler = MinMaxScaler()
    y_train_scaler = MinMaxScaler()
    y_test_scaler = MinMaxScaler()

    # Fit the scaler for the Training Data
    x_train_scaler.fit(X_train)
    y_train_scaler.fit(y_train)

    # Scale the training data
    X_train = x_train_scaler.transform(X_train)
    y_train = y_train_scaler.transform(y_train)

    # Fit the scaler for the Testing Data
    x_test_scaler.fit(X_test)
    y_test_scaler.fit(y_test)

    # Scale the y_test data
    X_test = x_test_scaler.transform(X_test)
    y_test = y_test_scaler.transform(y_test)
    return X_train, X_test, y_train, y_test, y_test_scaler


def testPickle():
    df = pd.read_csv('./Data/AAPL.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)
    df["Pct_change"] = df["Adj Close"].pct_change()
    # Drop null values
    df.dropna(inplace=True)
    df.head()
    window_size = 2
    feature_col_number1 = 4
    feature_col_number2 = 6
    feature_col_number3 = 7
    target_col_number = 4
    X, y = window_data(df, window_size, feature_col_number1, feature_col_number2, feature_col_number3,
                       target_col_number)
    X_train, X_test, y_train, y_test, y_test_scaler = splitData(X, y)
    pickled_model = pickle.load(open('./pickleFiles_sentiment/Stock_XGB.pkl', 'rb'))
    print(pickled_model.predict(X_test))


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


lasso_params = {"alpha": [0.01, 0.1, 1, 10, 100, 1000]}
params_SVR = {"C": [0.001, 0.01, 0.1, 1, 10], "epsilon": [0.01, 0.1, 1, 10]}

models = {'LinearRegression': LinearRegression(), 'Lasso': GridSearchCV(Lasso(), param_grid=lasso_params),
          'Ridge': GridSearchCV(Ridge(), param_grid=lasso_params), 'SVR': GridSearchCV(SVR(
        kernel='linear'), param_grid=params_SVR), 'XGB': XGBRegressor(objective='reg:squarederror', n_estimators=1000),
          'RandomForest': RandomForestRegressor(n_estimators=200, min_samples_split=50, random_state=1)}


def plotData(stocks, modelName):
    print("plotting")
    plt.plot(stocks.index, stocks['Real'])
    plt.plot(stocks.index, stocks['Adj Close'])
    plt.title("This is the results of the " + modelName)
    plt.legend(['Real', 'Predicted'])
    plt.show()


def convertToPickle(model):
    pickle.dump(model, open('pickleFiles_sentiment/Stock_XGB.pkl', 'wb'))


def runStockPredictionSentiment(modelName, getSignal, models=models):
    print("Models has just started")
    model = models[modelName]
    df = pd.read_csv('./Data/AAPL.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)
    df["Pct_change"] = df["Adj Close"].pct_change()
    # Drop null values
    df.dropna(inplace=True)
    df.head()
    window_size = 2
    feature_col_number1 = 4
    feature_col_number2 = 6
    feature_col_number3 = 7
    target_col_number = 4
    X, y = window_data(df, window_size, feature_col_number1, feature_col_number2, feature_col_number3,
                       target_col_number)
    X_train, X_test, y_train, y_test, y_test_scaler = splitData(X, y)
    predictions, RMSE, R_Squared = getPredictions(model, X_train, X_test, y_train, y_test)
    # convertToPickle(model)
    print("RMSE: " + str(RMSE))
    print("R_squared: " + str(R_Squared))
    predicted_prices = y_test_scaler.inverse_transform(predictions.reshape(-1, 1))
    real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))
    stocks, results, output_df = convertToPandas(df, real_prices, predicted_prices)

    if getSignal:
        frames = RSI_Calc(output_df)
        buyingsignals, sellingdates = getSignals(frames)
        profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
            buyingsignals].Open.values

        wins = [i for i in profits if i > 0]
        winning_rate = len(wins) / len(profits)

        print(winning_rate)

        return predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate
    return predicted_prices, real_prices, stocks, RMSE, R_Squared

def runStockPredictionSentimentWithSavedModel(modelName, getSignal):
    print("Loading saved model...")
    df = pd.read_csv('../Data/AAPL.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)
    df["Pct_change"] = df["Adj Close"].pct_change()
    # Drop null values
    df.dropna(inplace=True)
    df.head()
    window_size = 2
    feature_col_number1 = 4
    feature_col_number2 = 6
    feature_col_number3 = 7
    target_col_number = 4
    X, y = window_data(df, window_size, feature_col_number1, feature_col_number2, feature_col_number3,
                       target_col_number)
    X_train, X_test, y_train, y_test, y_test_scaler = splitData(X, y)
    predictions, RMSE, R_Squared = getPredictionsWithSavedModel(modelName, X_test, y_test)
    print("RMSE: " + str(RMSE))
    print("R_squared: " + str(R_Squared))
    predicted_prices = y_test_scaler.inverse_transform(predictions.reshape(-1, 1))
    real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))
    stocks, results, output_df = convertToPandas(df, real_prices, predicted_prices)

    if getSignal:
        frames = RSI_Calc(output_df)
        buyingsignals, sellingdates = getSignals(frames)
        profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
            buyingsignals].Open.values

        wins = [i for i in profits if i > 0]
        winning_rate = len(wins) / len(profits)

        print(winning_rate)

        return predicted_prices, real_prices, output_df, frames, buyingsignals, sellingdates, winning_rate
    return predicted_prices, real_prices, stocks, RMSE, R_Squared

# runStockPredictionSentiment('XGB', False)
# testPickle()
# runStockPredictionSentiment('RandomForest', False)
# pickled_model = pickle.load(open('./pickleFiles/Stock_LinearRegression.pkl', 'rb'))
