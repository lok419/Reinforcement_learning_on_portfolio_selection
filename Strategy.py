import pandas as pd
from universal import tools
from universal import algos
from datetime import datetime
from Portfolio import Portfolio
import random
import pickle
import logging
import numpy as np
# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
import matplotlib.pyplot as plt
import Performance

# It produces benchmarking strategy

with open("ticker/sp500tickers_All.pickle",'rb') as file:
    tickers = random.sample(pickle.load(file),5)


P = Portfolio(tickers, '2013-01-02', '2013-12-31', mode='train')
date = P.df_close.index[P.start_index:]

RMR = algos.RMR()
RMR_return = RMR.run(P.df_close.iloc[P.start_index:])
RMR_return.plot(weights=False, assets=False, ucrp=True, logy=False)

OLMAR = algos.OLMAR()
OLMAR_return = OLMAR.run(P.df_close.iloc[P.start_index:])
OLMAR_return.plot(weights=False, assets=False, ucrp=True, logy=False)

PAMR = algos.PAMR()
PAMR_return = PAMR.run(P.df_close.iloc[P.start_index:])
PAMR_return.plot(weights=False, assets=False, ucrp=True, logy=False)

Anticor = algos.Anticor()
Anticor_return = Anticor.run(P.df_close.iloc[P.start_index:])
Anticor_return.plot(weights=False, assets=False, ucrp=True, logy=False)


plt.show()
