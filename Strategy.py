from universal import algos
from Portfolio import Portfolio
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
import matplotlib.pyplot as plt

# It produces benchmarking strategy
# group 1
sector = "Information Technology"
tickers = ['AAPL', 'AMAT', 'AMD', 'CSCO', 'EBAY', 'GLW', 'HPQ', 'IBM', 'INTC', 'KLAC', 'MSFT', 'MU', 'NVDA', 'QCOM', 'TXN']

# # group 2
# sector = "Consumer Discretionary"
# tickers = ['AZO', 'BBY', 'DHI', 'F', 'GPS', 'GRMN', 'HOG', 'JWN', 'MAT', 'MCD', 'NKE', 'SBUX', 'TJX', 'TWX', 'YUM']
#
# group 3
# sector = "Industrials"
# tickers = ['BA', 'CAT', 'CTAS', 'EMR', 'FDX', 'GD', 'GE', 'LLL', 'LUV', 'MAS', 'MMM', 'NOC', 'RSG', 'UNP', 'WM']

P = Portfolio(tickers, '2016-01-04', '2016-12-31', mode='train')
date = P.df_close.index[P.start_index:]

RMR = algos.RMR()
RMR_return = RMR.run(P.df_close.iloc[P.start_index:])
RMR_return.plot(weights=True, assets=False, ucrp=True, logy=False)

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
