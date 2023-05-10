import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn import metrics
import numpy as np
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.backends.backend_tkagg as tkagg
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

class ThirdFrame(tk.Frame):
    def __init__(self, controller, master=None):
        super().__init__(master)
        self.controller = controller

        # create a label
        label = ttk.Label(self, text="This is the second page", font=('Arial', 24))
        label.pack(pady=50)
        # create a button to go back to the first page
        button = ttk.Button(self, text="Go back", command=self.show_first_page)
        button.pack()

        # create a button to show the graph
        button = ttk.Button(self, text="Show Results", command=self.show_graph)
        button.pack()

        # create a figure to hold the plot
        self.figure = plt.figure(figsize=(6, 4), dpi=100)

        # create a canvas to display the figure
        self.canvas = tkagg.FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack()

    def show_first_page(self):
        self.pack_forget()
        self.controller.canvas.pack()

    def show_graph(self):
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

        # Testing the functions

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
        features = ['Open', 'High', 'Low']
        target_col = 'Adj Close'
        X_train, X_test, y_train, y_test, y_train_dates, y_test_dates = splitData(stock_clean, 0.8, features,
                                                                                  target_col)
        X_train, X_test = scaleData(X_train, X_test)
        output = getPrediction(models['LinearRegression'], X_train, X_test, y_train)

        output_df = convertPredictionsIntoPandas(output, y_test_dates, X_test, features)
        print(output_df)
        print(mean_squared_error(y_test, output, squared=False))
        print("Test", y_test)
        print("Predicted", output)
        print(output_df)

        # Get Signals from prediction part

        frames = RSI_Calc(output_df)
        buyingsignals, sellingdates = getSignals(frames)

        # plt.figure(figsize=(12, 5))
        # plt.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
        # plt.plot(frames['Adj Close'], alpha=0.7)
        # plt.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
        # plt.plot(frames['Adj Close'], alpha=0.7)
        # plt.show()

        profits = (frames.loc[sellingdates].Open.values - frames.loc[buyingsignals].Open.values) / frames.loc[
            buyingsignals].Open.values

        wins = [i for i in profits if i > 0]
        winning_rate = len(wins) / len(profits)

        print(winning_rate)

        ax = self.figure.add_subplot(111)
        ax.scatter(frames.loc[buyingsignals].index, frames.loc[buyingsignals]['Adj Close'], marker='^', c='g')
        ax.plot(frames['Adj Close'], alpha=0.7)
        ax.scatter(frames.loc[sellingdates].index, frames.loc[sellingdates]['Adj Close'], marker='^', c='r')
        ax.plot(frames['Adj Close'], alpha=0.7)
        # plt.show()
        # ax.plot(X_train, y_train.ravel())
        # ax.set_title('Predictions')
        self.canvas.draw()

        # create a label
        strLabel1 = 'Winning Rate: ' + str(winning_rate)
        label = ttk.Label(self, text=strLabel1)
        label.pack()
        # strLabel2='R-squared :'+ str(metrics.r2_score(y_test, predicted))
        # label2 = ttk.Label(self, text=strLabel2)
        # label2.pack()


