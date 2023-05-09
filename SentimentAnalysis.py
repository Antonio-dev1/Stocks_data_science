import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np
from Signal_Algorithm import RSI_Calc,getSignals

df = pd.read_csv('./data/AAPL.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)
df["Pct_change"] = df["Adj Close"].pct_change()
# Drop null values
df.dropna(inplace = True)
df.head()


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



# Predict Closing Prices using a 3 day window of previous closing prices
window_size = 3

# Column index 0 is the `Adj Close` column
# Column index 1 is the `ts_polarity` column
# Column index 2 is the `twitter_volume` column
feature_col_number1 = 4
feature_col_number2 = 6
feature_col_number3 = 7
target_col_number = 4
X, y = window_data(df, window_size, feature_col_number1, feature_col_number2, feature_col_number3, target_col_number)



# Use 70% of the data for training and 30% for testing
X_split = int(0.3* len(X))
y_split = int(0.3 * len(y))

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


# Create the XG Boost regressor instance
#model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)

#model = LinearRegression()
lasso = Lasso()
params = {"alpha":[0.001 , 0.01 , 0.1 , 1, 10 ,100 , 1000]}
model = GridSearchCV(lasso , param_grid=params)

# Fit the model
model.fit(X_train, y_train.ravel())



predicted = model.predict(X_test)
# Evaluating the model
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
print('R-squared :', metrics.r2_score(y_test, predicted))


predicted_prices = y_test_scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))
# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Adj Close": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ])

results = pd.DataFrame({
    "Adj Close": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ])

originaldata = df.loc[stocks.index , ['Close' , 'Open' , 'High' , 'Low']]

output_df = pd.concat([originaldata  , results] , axis=1)
print(model.best_params_)
plt.plot(stocks.index , stocks['Real'])
plt.plot(stocks.index , stocks['Adj Close'])
plt.legend(['Real' , 'Predicted'])
plt.show()

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




