try:
    import quandl
except ImportError:
    pass

import pandas as pd
import os
import numpy as np
import random
import sys
import datetime
import Data
import Indicator
from pandas.tseries.offsets import BDay

class Portfolio(object):

    def __init__(self,tickers, start, end, mode='train'):
        self.tickers = tickers[:]
        self.tickers_num = len(self.tickers)
        self.start = start
        self.before_start = (datetime.datetime.strptime(start, '%Y-%m-%d') - BDay(200)).strftime('%Y-%m-%d')
        self.end = end
        self.mode = mode
        self.consistent = True

        # fundamental indicator
        self.df_close = self.extract_data(self.tickers, self.start, self.end, label='Adj. Close')
        self.df_high = self.extract_data(self.tickers, self.start, self.end, label='Adj. High').fillna(method='ffill')
        self.df_low = self.extract_data(self.tickers, self.start, self.end, label='Adj. Low').fillna(method='ffill')
        self.df_open = self.extract_data(self.tickers, self.start, self.end, label='Adj. Open').fillna(method='ffill')
        self.df_volume = self.extract_data(self.tickers, self.start, self.end, label='Volume').fillna(method='ffill')

        # technical indicator
        self.df_roc = Indicator.roc(self.df_close).fillna(method='ffill')
        self.df_macd = Indicator.macd(self.df_close).fillna(method='ffill')
        self.df_ma5 = Indicator.ma(self.df_close, 5).fillna(method='ffill')
        self.df_ma10 = Indicator.ma(self.df_close,10).fillna(method='ffill')
        self.df_ema20 = Indicator.ema(self.df_close,20).fillna(method='ffill')
        self.df_sr20 = Indicator.sharp_ratio(self.df_close,20).fillna(method='ffill')

        try:
            self.start_index = np.where(self.df_close.index == start)[0][0]
        except IndexError:
            pass

        self.df_normalized = (self.df_open.shift(-1) / self.df_open).fillna(1)            # next day's price / today's price
        self.portfolio_weights = list()
        self.portfolio_return = list([1])
        self.portfolio_UCRP = list([1])
        self.portfolio_UBHP = list([1])
        self.transaction_factor = 0.0025

        print('Created Portfolio: {}'.format(tickers))
        print('Number of assets: {}'.format(self.tickers_num))
        print('Included Cash Bias')
        if mode == 'train':
            print('Training period: {} to {}'.format(start,end))
        elif mode == 'valid':
            print('Validating period: {} to {}'.format(start,end))
        elif mode == 'test':
            print('Testing period: {} to {}'.format(start,end))

    def calculate_return(self, weight, last_weight, step ):
        # make sure sum of weight equal to 1
        transaction_cost = 1 - self.transaction_factor * np.sum(np.abs(weight[:-1] - last_weight))
        future_price = np.append(self.df_normalized.iloc[step + self.start_index].values, [1])
        day_return = np.dot(future_price, weight) * transaction_cost
        self.portfolio_return.append(day_return)
        self.portfolio_weights.append(weight)

        return day_return, future_price

    # clear the portfolio and return when starting a new episode
    def clear(self):
        self.portfolio_weights = list()
        self.portfolio_return  = list([1])


    # extract data individually, and store to data/stock
    def import_from_quandl(self, symbol, start, end):
        if os.path.exists('./data/stock/{}.csv'.format(symbol)):
            print('Read from stored data:', symbol)
        else:
            Data.download_data(symbol)
        df = pd.read_csv('./data/stock/{}.csv'.format(symbol), header=0, parse_dates=True, index_col=0)[self.before_start : end]
        if df.empty:
            print("Historical data of {} does not cover period from {} to {}".format(symbol, start, end))
            self.consistent = False
        elif df.index[0] != datetime.datetime.strptime(self.before_start,'%Y-%m-%d'):
            print("Historical data of {} does not cover period from {} to {}".format(symbol, start, end))
            self.consistent = False
        return df

    # extract a group of data with same label according to tickers
    def extract_data(self, tickers, start, end, label):
        main_df = pd.DataFrame()
        for ticker in tickers:
            df = self.import_from_quandl(ticker, start, end)
            df.rename(columns={label:ticker}, inplace=True)
            df = df[[ticker]]
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')
        return main_df

    # randomly shuffle the assets order
    def shuffle(self):
        random.shuffle(self.tickers)

        # fundamental indicator
        self.df_close = self.df_close[self.tickers]
        self.df_open = self.df_open[self.tickers]
        self.df_high = self.df_high[self.tickers]
        self.df_low = self.df_low[self.tickers]
        self.df_volume = self.df_volume[self.tickers]

        # technical indicator
        self.df_roc = self.df_roc[self.tickers]
        self.df_normalized = self.df_normalized[self.tickers]


    # Uniform Constant Rebalance Portfolio
    def UCRP(self):
        weight = np.array([1 for _ in range(self.tickers_num)]) / self.tickers_num
        for j in range(self.start_index, len(self.df_normalized) - 1):
            day_return = np.dot(self.df_normalized.iloc[j], weight)
            self.portfolio_UCRP.append(day_return)

    # Uniform Buy and Hold Portfolio
    def UBHP(self):
        weight = np.array([1 for _ in range(self.tickers_num)]) / self.tickers_num
        for j in range(self.start_index, len(self.df_normalized) - 1):
            day_return = np.dot(self.df_normalized.iloc[j], weight)
            self.portfolio_UBHP.append(day_return)
            weight *= self.df_normalized.iloc[j]
            weight = weight / sum(weight)



