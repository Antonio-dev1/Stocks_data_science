import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from scipy import stats
from scipy.stats import skew,kurtosis
from datetime import datetime
import matplotlib as mpl
from functools import reduce
import datetime as dt
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
import yfinance as yf
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2023, 1, 11)

tickers = ['AAPL']

apple_df = yf.download(tickers , start=start , end=end)

apple_df.head()

plt.plot(apple_df.index , apple_df['Close'])
plt.show()

X = apple_df.drop(['Adj Close', 'Close'] , axis =1 ).values

y = apple_df['Adj Close'].values

split = int(0.8 * len(apple_df))
train_data = apple_df.iloc[:split]
test_data = apple_df.iloc[split:]

scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

target_col = 'Adj Close'
n_steps = 60
n_features = 5

X_train = []
y_train = []
for i in range(n_steps, len(train_data_scaled)):
    X_train.append(train_data_scaled[i-n_steps:i, :n_features])
    y_train.append(train_data_scaled[i, apple_df.columns.get_loc(target_col)])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []
for i in range(n_steps, len(test_data_scaled)):
    X_test.append(test_data_scaled[i-n_steps:i, :n_features])
    y_test.append(test_data_scaled[i, apple_df.columns.get_loc(target_col)])
X_test, y_test = np.array(X_test), np.array(y_test)

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32)

y_pred = model.predict(X_test).flatten()
print(y_pred.shape)
# rescale predictions and actual values
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Test RMSE: %.3f' % rmse)

dates_train = apple_df.index[:split]
dates_test = apple_df.index[split:]

print(y_pred)
print(y_test)
plt.plot( y_pred)
plt.plot( y_test)
plt.legend(['Predictions' , 'Actual'])

#Bad model with Scaling , good model with Min Max Scaler
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

target_col = 'Adj Close'
n_steps = 60
n_features = 5

X_train = []
y_train = []
for i in range(n_steps, len(train_data_scaled)):
    X_train.append(train_data_scaled[i-n_steps:i, :n_features])
    y_train.append(train_data_scaled[i, apple_df.columns.get_loc(target_col)])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []
for i in range(n_steps, len(test_data_scaled)):
    X_test.append(test_data_scaled[i-n_steps:i, :n_features])
    y_test.append(test_data_scaled[i, apple_df.columns.get_loc(target_col)])
X_test, y_test = np.array(X_test), np.array(y_test)

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32)

y_pred = model.predict(X_test).flatten()
print(y_pred.shape)
# rescale predictions and actual values
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Test RMSE: %.3f' % rmse)

print(y_pred)
print(y_test)
plt.plot(y_pred)
plt.plot( y_test)
plt.legend(['Predictions' , 'Actual'])
plt.show()

