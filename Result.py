import pickle
import matplotlib.pyplot as plt
from matplotlib import style
from cycler import cycler
import pandas
from Portfolio import Portfolio
import Performance
import os
import Ssh
import numpy as np
from universal import algos
import logging

# plot settings
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
style.use('ggplot')
plt.rcParams['grid.color'] = 'lightgrey'
plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['axes.prop_cycle'] = cycler(color=[u'#1f77b4',u'#d62728',u'#ff7f0e',u'#7f7f7f',u'#2ca02c',u'#98df8a',u'#aec7e8',u'#ff9896',u'#9467bd',u'#c5b0d5',
                                                u'#8c564b',u'#c49c94',u'#e377c2',u'#f7b6d2',u'#ffbb78',u'#c7c7c7',u'#bcbd22',u'#dbdb8d',u'#17becf',u'#9edae5'])

class Result(object):

    def __init__(self,filename):

        P = self.read_pickle(filename)
        self.tickers = P.tickers
        self.start_index = P.start_index
        self.date = P.df_close.index[self.start_index:]
        self.weights = P.portfolio_weights
        self.close = P.df_close
        self.model = ['model','UCRP','UBHP','RMR','OLMAR','PAMR','Anticor']
        self.model_return = []
        self.model_return.append(P.portfolio_return)

        # Passive trading strategy (UCRP and UBHP)
        self.model_return.append(P.portfolio_UCRP)
        self.model_return.append(P.portfolio_UBHP)

        # Robust median reversion strategy
        RMR = algos.RMR()
        RMR_return = RMR.run(self.close.iloc[self.start_index:])
        self.model_return.append(RMR_return.r)

        # Online moving average reversion strategy
        OLMAR = algos.OLMAR()
        OLMAR_return = OLMAR.run(self.close.iloc[self.start_index:])
        self.model_return.append(OLMAR_return.r)

        # Passive aggressive mean reversion strategy
        PAMR = algos.PAMR()
        PAMR_return = PAMR.run(self.close.iloc[self.start_index:])
        self.model_return.append(PAMR_return.r)

        # Anticor (anti-correlation)
        Anticor = algos.Anticor()
        Anticor_return = Anticor.run(self.close.iloc[self.start_index:])
        self.model_return.append(Anticor_return.r)

    def read_pickle(self, filename):
        if not (os.path.exists('./result/{}'.format(filename))):
            # Ssh.getall('./result','./result')
            Ssh.get(local = './result/{}'.format(filename), server = './result/{}'.format(filename))
        with open('./result/'+filename,'rb') as file:
            P = pickle.load(file)
        return P

    def plot_weights(self):
        plt.title('Weights of assets')
        plt.xlabel('Date')
        plt.ylabel('Weights')
        plt.plot(self.date, self.weights)
        plt.legend(np.concatenate([self.tickers,['Cash']]))

    def plot_price(self):
        plt.title('Adjusted Close')
        plt.xlabel('Date')
        plt.plot(self.close.iloc[self.start_index:])
        plt.legend(self.tickers)

    def plot_porfolio_return(self):
        plt.title('Portfolio Return')
        plt.ylabel('Cumulative Return')
        plt.xlabel('Date')
        # plt.ylim((0.75,2.75))

        for i in range(len(self.model)):
            if i < 4:
                plt.plot( self.date, Performance.cummulative_return(self.model_return[i]) , linestyle = '-')
            else:
                plt.plot( self.date, Performance.cummulative_return(self.model_return[i]) , linestyle = '--')
        plt.legend(self.model, loc=2)

    def plot_epsiode_reward(self,filename):
        with open('./result/'+ filename,'rb') as file:
            self.epsiode_reward = pickle.load(file)
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        plt.title('Episode Performance')
        plt.plot(self.epsiode_reward)

    def plot_return_and_weights(self):
        ax1 = plt.subplot(211)
        plt.title('Portfolio Return')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.title('Weights of assets')
        ax1.plot( self.date, Performance.cummulative_return(self.model_return[0]) )
        ax1.tick_params(labelbottom='off')
        ax2.plot( self.date, self.weights)
        ax2.set_xlabel('Date')
        ax2.legend(np.concatenate([self.tickers,['Cash']]))

    def plot_price_and_weights(self):
        ax1 = plt.subplot(211)
        plt.title('Adjusted Close (normalized)')
        ax2 = plt.subplot(212, sharex=ax1)
        plt.title('Weights of assets')
        ax1.plot(self.close.iloc[self.start_index:] / self.close.iloc[self.start_index])
        ax1.legend(self.tickers)
        ax1.tick_params(labelbottom='off')
        ax2.plot(self.date, self.weights)
        ax2.set_xlabel('Date')
        ax2.legend(np.concatenate([self.tickers,['Cash']]), loc=5)

    def perforamnce(self):
        max_drawdown = []
        sharpe_ratio = []
        cumulative_return = []
        volatility = []
        performance_indicator = ['Cumulative Return','Max Drawdown','Sharpe Ratio','Volatility']
        for i, model in enumerate(self.model):
            max_drawdown.append(Performance.max_drawdown(self.model_return[i]))
            sharpe_ratio.append(Performance.sharpe_ratio(self.model_return[i]))
            cumulative_return.append(Performance.cummulative_return(self.model_return[i])[-1])
            volatility.append(Performance.volatility(self.model_return[i]))
        table = pandas.DataFrame(np.array([cumulative_return,max_drawdown,sharpe_ratio, volatility]), performance_indicator, self.model)
        print("")
        print("Trading Period: {} - {}".format(self.date[0].strftime('%Y-%m-%d'),self.date[-1].strftime('%Y-%m-%d')))
        print(table)
        print("")
        print("Highest Cumulative Return: {}".format(self.model[np.argmax(cumulative_return).item()]))
        print("Lowest Maximum Drawdown: {}".format(self.model[np.argmin(max_drawdown).item()]))
        print("Highest Sharpe Ratio: {}".format(self.model[np.argmax(sharpe_ratio).item()]))
        print("Lowest Volatility: {}".format(self.model[np.argmin(volatility).item()]))

    def plot_all(self):
        plt.figure(1)
        self.plot_weights()
        plt.figure(2)
        self.plot_price()
        plt.figure(3)
        self.plot_porfolio_return()
        plt.figure(4)
        self.plot_epsiode_reward('Epsiode_Performance_{}_{}_{}.pickle'.format(self.P.mode, self.P.start, self.P.end))
        plt.figure(5)
        self.plot_return_and_weights()
        plt.figure(6)
        self.plot_price_and_weights()
        self.perforamnce()
        plt.show()


if __name__ == "__main__":
    filename = "CNN_result.pickle"
    result = Result(filename)
    result.plot_all()
    plt.show()