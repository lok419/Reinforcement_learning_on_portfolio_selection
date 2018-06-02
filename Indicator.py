import pandas as pd
import numpy as np

# price should be in form of dataframe

# Moving Average Convergence / Divergence
def macd(price):
    return price.ewm(span=12,min_periods=0,adjust=True,ignore_na=False).mean() / price.ewm(span=26,min_periods=0,adjust=True,ignore_na=False).mean()

# rate of change
def roc(price):
    return (price / price.shift(1)).fillna(1)

# Simple moving average
def ma(price, day):
    return price.rolling(window=day,min_periods=1,center=False).mean()

# Exponential Moving Average
def ema(price, day):
    return price.ewm(span=day, min_periods=0,adjust=True,ignore_na=False).mean()

def volatility(price, day):
    return price.rolling(window=day,min_periods=1,center=False).std()

def sharp_ratio(price, day):
    price = np.log((price / price.shift(1)).fillna(1))
    sliding_window = price.rolling(window=day,min_periods=1,center=False)

    # Risk free rate for sharpe ratio (Three-month U.S. Treasury bill)
    risk_free_rate = np.log(1.0148) / 60

    return np.sqrt(sliding_window.count()) * (sliding_window.mean() - risk_free_rate)  / sliding_window.std()
