import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
import yfinance as yf

start = dt.datetime(2010, 1, 1)
end = dt.datetime(2017, 1, 11)

tickers = ['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT']
df_finance = yf.download(tickers, start=start, end=end)
apple_df = yf.download('AAPL' , start=start)
print(df_finance)
stocks_adj_close = df_finance['Adj Close']

# Get the correlation betwee adjustment close
retscomp = stocks_adj_close.pct_change()
corr = retscomp.corr()

pd.plotting.scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));

# To calculate the risk we are trying to (In the plot each point represents a stock)
plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
    plt.annotate(
        label,
        xy=(x, y), xytext=(20, -20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Setting the new highLow average change in a day and different features to predict the price
dfreg = apple_df.loc[:, ['Adj Close', 'Volume']]

dfreg['HL_PCT'] = (apple_df['High'] - apple_df['Low']) / apple_df['Close'] * 100.0
dfreg['PCT_change'] = (apple_df['Close'] - apple_df['Open']) / apple_df['Open'] * 100.0

dfreg.head()

# Preprocessing the data
dfreg.fillna(value=-9999, inplace=True)
# Getting the amount of forecast data's length(How many points do we want to forecast)
forecast_out = int(math.ceil(0.01 * len(dfreg)))

# The label to predict is adj Close
prediction_col = 'Adj Close'
dfreg['label'] = dfreg[prediction_col].shift(-forecast_out)

dfreg.head()

X = dfreg.drop('label', 1).values

print(X)
X = preprocessing.scale(X)

X = dfreg.drop('label', 1).values

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = dfreg['label'].values
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

legreg = LinearRegression(n_jobs=-1)
legreg.fit(X_train, y_train)

legreg.score(X_test, y_test)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
legreg_cv = cross_val_score(legreg, X, y, cv=kf)
print(legreg_cv)

forecasts = legreg.predict(X_lately)
y_true = y[-forecast_out:]
forecasts
print(forecasts)
mse = mean_squared_error(y_true, forecasts, squared=True)
print(mse)

param_grid = {'alpha': [0.1, 0.5, 1, 10, 20, 30, 50, 70, 90]}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)

# Ridge better MSE and better accuracy score
print(ridge_cv.best_params_, ridge_cv.best_estimator_)
predictions = ridge_cv.predict(X_lately)
error_ridge = mean_squared_error(y_true, predictions, squared=True)
print(error_ridge)
ridge_cv.score(X_test, y_test)
