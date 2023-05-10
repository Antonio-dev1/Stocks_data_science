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


class SecondFrame(tk.Frame):
    def __init__(self, controller, master=None):
        super().__init__(master)
        self.controller = controller

        # create a label
        label = ttk.Label(self, text="This page to predict Sentiment Analysis", font=('Arial', 24))
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
        df = pd.read_csv('./data/AAPL.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)
        df["Pct_change"] = df["Adj Close"].pct_change()
        # Drop null values
        df.dropna(inplace=True)
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

        # Predict Closing Prices using a 3 day window of previous closing prices
        window_size = 3

        # Column index 0 is the `Adj Close` column
        # Column index 1 is the `ts_polarity` column
        # Column index 2 is the `twitter_volume` column
        feature_col_number1 = 4
        feature_col_number2 = 6
        feature_col_number3 = 7
        target_col_number = 4
        X, y = window_data(df, window_size, feature_col_number1, feature_col_number2, feature_col_number3,
                           target_col_number)

        # Use 70% of the data for training and 30% for testing
        X_split = int(0.7 * len(X))
        y_split = int(0.7 * len(y))

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
        model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)

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
            "Predicted": predicted_prices.ravel()
        }, index=df.index[-len(real_prices):])
        stocks.head()
        ax = self.figure.add_subplot(111)
        ax.plot(X_train, y_train.ravel())
        ax.set_title('Predictions')
        self.canvas.draw()

        # create a label
        strLabel1 = 'Accuracy: ' + str(np.sqrt(metrics.mean_squared_error(y_test, predicted)))
        label = ttk.Label(self, text=strLabel1)
        label.pack()
        strLabel2='R-squared :'+ str(metrics.r2_score(y_test, predicted))
        label2 = ttk.Label(self, text=strLabel2)
        label2.pack()



