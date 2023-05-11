import pandas as pd
import numpy as np
import math
from scipy import stats
import yfinance as yf
import matplotlib.pyplot as plt

ticker = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

tickers = ticker.Symbol.to_list()

tickers = [i.replace('.', '-') for i in tickers]

# Function that calculates RSI for a single trading day
def RSI_Calc(df):
    # Gets the moving average
    df['MA200'] = df['Adj Close'].rolling(window=120).mean()
    # Up and down price moves and relative returns
    df['price change'] = df['Adj Close'].pct_change()
    # Take the daily return if postive else just take zero
    df['Upmove'] = df['price change'].apply(lambda x: x if x > 0 else 0)
    # Taking the negative returns else take it to be zeor
    df['Downmove'] = df['price change'].apply(lambda x: abs(x) if x < 0 else 0)
    # Getting the avergae Up, span of 19 days taking the down column
    df['avg up'] = df['Upmove'].ewm(span=19).mean()
    # Getting the avergae down, span of 19 days taking the down column
    df['avg down'] = df['Downmove'].ewm(span=19).mean()
    # na caused from the rolling window and the pct change
    df = df.dropna()

    df['RS'] = df['avg up'] / df['avg down']
    # Get the RSI value for every trading day
    df['RSI'] = df['RS'].apply(lambda x: 100 - (100 / (x + 1)))
    df.loc[(df['Adj Close'] > df['MA200']) & (df['RSI'] < 30), 'Buy'] = 'Yes'
    df.loc[(df['Adj Close'] < df['MA200']) | (df['RSI'] > 30), 'Buy'] = 'No'

    return df


# Gets the buying signals buying on the next days open
# Subsequent trading days if not breaking the loop
def getSignals(df):
    buying_dates = []
    selling_dates = []
    for i in range(len(df) - 11):
        if "Yes" in df['Buy'].iloc[i]:
            buying_dates.append(df.iloc[i + 1].name)
            for j in range(1, 11):
                if df['RSI'].iloc[i + j] > 40:
                    selling_dates.append(df.iloc[i + j + 1].name)
                    break  # If RSI exceeding 40 stock too risky exit trade
                elif j == 10:
                    selling_dates.append(df.iloc[i + j + 1].name)
    return buying_dates, selling_dates


# frames = RSI_Calc(tickers[0])
# buyingsignals, sellingdates = getSignals(frames)
#
# plt.figure(figsize=(12, 5))
# plt.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
# plt.plot(frames['Adj Close'], alpha=0.7)
#
# profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
#     buyingsignals].Open.values
#
#
# wins = [i for i in profits if i > 0]
# winning_rate = len(wins) / len(profits)
#
# buying_matrix = []
# profits_matrix = []
#
# for i in range(len(tickers) - 400):
#     frame = RSI_Calc(tickers[i])
#     buy, sell = getSignals(frame)
#     profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
#         buyingsignals].Open.values
#     buying_matrix.append(buy)
#     profits_matrix.append(profits)
#
# allProfits = []
# for i in profits_matrix:
#     for e in i:
#         allProfits.append(e)
#
# wins = [i for i in allProfits if i > 0]
#
# winnings = len(wins) / len(allProfits)
#
# plt.hist(allProfits, bins=200)
# plt.show()
