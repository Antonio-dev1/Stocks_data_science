# Asset Management based on Stock Market and Cryptocurrency Predictions
This project consists of predicting the prices of different stocks and based on those predictions getting the signals to buy/sell.
The winning rate is around 50% with accuracy of 98%.

The models used are XGBBoost Regressor, LSTM, Linear Regression, Lasso and Ridge.
Additionally, a model included sentiment analysis from data used from kaggle

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the packages from the requirements.txt file.


## Usage
Run the main.py file in the gui folder
All models are ran in the gui, main, we are predicting live.
```commandline
cd Stocks_data_science/gui
main.py
```

## Pickle
.pkl files are some saved models but since we are running live data we have to train the model multiple times on the new data
