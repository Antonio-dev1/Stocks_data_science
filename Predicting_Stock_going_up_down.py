
import yfinance as yf
import pandas as pd

ticker = ['^GSPC']
SP_500 = yf.download(ticker, start='2011-01-01')

# Shift the close by one day show columns of tommrows price
SP_500["tommorow"] = SP_500["Close"].shift(-1)

SP_500

SP_500["target"] = (SP_500["tommorow"] > SP_500["Close"]).astype(int)

# Trying to be able to predict data from the if the price increased the next day not decreased
SP_500["target"]

from sklearn.ensemble import RandomForestClassifier

# Random forest do not tend to overfit and can draw conclusions from non linear data and relationships
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
# Split this correct since you can not run a KF since we do not want to have a leakage
train = SP_500.iloc[:-100]
test = SP_500.iloc[-100:]
predictors = ["Close", "Volume", "High", "Low"]
model.fit(train[predictors], train["target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

preds

precision_score(test["target"], preds)


# Back testing system
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["target"], preds], axis=1)
    return combined


# Take 10 years of data and train your first model with 10 years of data, get predictions for a lot of different years
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


predictions = backtest(SP_500, model, predictors)

# Trying the SVC model
precision_score(predictions["target"], predictions["Predictions"])
predictions["target"].value_counts() / predictions.shape[0]

horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = SP_500.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    SP_500[ratio_column] = SP_500["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    SP_500[trend_column] = SP_500.shift(1).rolling(horizon).sum()["target"]

    new_predictors += [ratio_column, trend_column]

SP_500 = SP_500.dropna()

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# Changing the threshold to make sure that the price actually goes up and it will reduce the number of days the price goes up
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["target"], preds], axis=1)
    return combined


