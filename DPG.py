import numpy as np
import tensorflow as tf
import pickle
from Portfolio import Portfolio
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from sys import exit
import os
import Data
import random


def gpu_settings():
    os.environ["CUDA_VISIBLE_DEVICES"] = '4' #use GPU with ID 4
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    K.set_learning_phase(1)
    return sess

def load_weights(mode, actor, ep):
    print("Now loading the weight")
    try:
        if mode == "train" or mode == "valid":
            actor.model.load_weights("./model/actormodel_{}_ep{}.h5".format("train", ep-1))
            print("Loaded actormodel_{}_ep{}.h5".format("train", ep-1))
        elif mode == "test":
            actor.model.load_weights("./model/actormodel_{}_ep{}.h5".format("valid", ep-1))
            print("Loaded actormodel_{}_ep{}.h5".format("valid", ep-1))
        else:
            raise Exception("Invalid mode")
        print("Loaded weight successfully ({})".format(ep-1))
    except:
        print("Cannot find the weight")

def save_weights_results(P, mode, actor, ep, performance):
    print("Now we save model")
    if mode == "train":
        actor.model.save_weights("./model/actormodel_{}_ep{}.h5".format(mode, ep), overwrite=True)
        print("Saved actormodel_{}_ep{}.h5".format(mode, ep))
        filename = 'Portfolio_{}_{}_{}_{}_{}_ep{}.pickle'.format(mode,P.start,P.end,'-'.join(P.tickers), performance, ep)
    else:
        if mode == "valid":
            actor.model.save_weights("./model/actormodel_{}_ep{}.h5".format(mode, ep-1), overwrite=True)
            print("Saved actormodel_{}_ep{}.h5".format(mode, ep-1))
        filename = 'Portfolio_{}_{}_{}_{}_{}_ep{}.pickle'.format(mode,P.start,P.end,'-'.join(P.tickers), performance, ep-1)

    with open("./result/{}".format(filename),"wb") as outfile:
        pickle.dump(P, outfile)
        print("Saved {}".format(filename) )

def save_epsoide_performance(mode, start, end, epsiode_reward):
    filename = 'Epsiode_Performance_{}_{}_{}'.format(mode,start,end)
    with open("./result/{}.pickle".format(filename),"wb") as outfile:
        pickle.dump(epsiode_reward, outfile)
        print("saved {}".format(filename))

def prepare_tickers(train_mode, mode, tickers, sector, start, end):

    # train mode 1: 10% of actual portfolio is replaced by other stocks
    # train mode 2: completely random portfolio
    # train mode 3: non random / noise portfolio

    if mode == "train":

        if train_mode == 1:
            print("Construct a portfolio with noise ({})".format(sector))
            rand_sample_num = int((len(tickers))*0.1)
            rand_tickers_num = np.random.randint(0,len(tickers),rand_sample_num)

            # create portfolio with noise (not actual one)
            train_tickers = tickers[:]

            if not os.path.exists("ticker/sp500tickers_{}_{}_{}.pickle".format(sector,start,end)):
                Data.generate_list(sector,start,end)
            with open("ticker/sp500tickers_{}_{}_{}.pickle".format(sector,start,end),"rb") as file:
                # randomly sample from sp500 but excluding original tickers
                rand_sample = random.sample([symbol for symbol in pickle.load(file) if symbol not in tickers], rand_sample_num)
                for ix in range(rand_sample_num):
                    train_tickers[rand_tickers_num[ix]] = rand_sample[ix]
            return train_tickers

        elif train_mode == 2:
            print("Construct a random portfolio ({})".format(sector))
            if not os.path.exists("ticker/sp500tickers_{}_{}_{}.pickle".format(sector,start,end)):
                Data.generate_list(sector,start,end)
            with open("ticker/sp500tickers_{}_{}_{}.pickle".format(sector,start,end),"rb") as file:
                train_tickers = random.sample(pickle.load(file), len(tickers))
            return train_tickers

        else:
            return tickers

    else:
        return tickers

def prepare_states(P,window_size):
    state = []
    df = np.array([P.df_close.values, P.df_open.values, P.df_high.values, P.df_low.values,
                   P.df_volume.values, P.df_roc.values, P.df_macd.values, P.df_ma5.values,
                   P.df_ma10.values, P.df_ema20.values, P.df_sr20.values], dtype='float')

    # if 1-1-2013 is the first day of trading, last 60 days (not include today) is the input for first day
    for j in range( P.start_index-1 , len(df[0])):
        temp = np.copy(df[:, j-window_size+1:j+1 , :])

        # to normalize the data
        # latest adjusted close price within sliding window
        # price_base = np.copy(temp[0,-1,:])
        for feature in [0,1,2,3,4,7,8,9]:
            for k in range(P.tickers_num):
                if temp[feature,-1,k] == 0:
                    temp[feature,:,k] /= temp[feature,-2,k]
                else:
                    temp[feature,:,k] /= temp[feature,-1,k]
        state.append(temp)
    return state

def experience_replay(actor, buff, BATCH_SIZE, mode,  state, step, future_price, last_action, cash_bias, prediction,
                      train_rolling_steps, test_rolling_steps, sess):

    # define the number of experience replay in each step
    if mode == "train":
        rolling_steps = train_rolling_steps
    else:
        rolling_steps = test_rolling_steps

    # Add historical data to replay buffer
    buff.add(state[step], future_price, last_action, prediction)

    # online mini-batch replay, for both test and train data
    for _ in range(rolling_steps):
        batch, current_batch_size = buff.getBatch(BATCH_SIZE)
        states = np.asarray([e[0] for e in batch])
        future_prices = np.asarray([e[1] for e in batch])
        last_actions = np.asarray([e[2] for e in batch])
        predictions = np.asarray([e[3] for e in batch])

        # to test the output between layers
        # test = sess.run(actor.test, feed_dict={
        #     actor.state: states,
        #     actor.last_action: last_actions,
        #     actor.future_price: future_prices,
        #     actor.cash_bias: np.array([[cash_bias] for _ in range(current_batch_size)]),
        #     actor.prediction: predictions
        # })
        # print(test[0])

        # Train only if the buffer filled with certain data
        if step > 10 :
            actor.train(states, last_actions, future_prices, np.array([[cash_bias] for _ in range(current_batch_size)]), predictions)



def Simulate(start, end, mode='train', ep_train=30, ep_start=0):

    # sector = ["Consumer Discretionary", "Consumer Staples", "Energy", "Financials", "Health Care", "Industrials",
    #           "Information Technology", "Materials","Real Estate","Telecommunication Services","Utilities"]

    # group 1
    # sector = "Information Technology"
    # tickers = ['AAPL', 'AMAT', 'AMD', 'CSCO', 'EBAY', 'GLW', 'HPQ', 'IBM', 'INTC', 'KLAC', 'MSFT', 'MU', 'NVDA', 'QCOM', 'TXN']

    # group 2
    # sector = "Consumer Discretionary"
    # tickers = ['AZO', 'BBY', 'DHI', 'F', 'GPS', 'GRMN', 'HOG', 'JWN', 'MAT', 'MCD', 'NKE', 'SBUX', 'TJX', 'TWX', 'YUM']
    #
    # group 3
    sector = "Industrials"
    tickers = ['BA', 'CAT', 'CTAS', 'EMR', 'FDX', 'GD', 'GE', 'LLL', 'LUV', 'MAS', 'MMM', 'NOC', 'RSG', 'UNP', 'WM']


    train_mode = 2
    BUFFER_SIZE = 200
    BATCH_SIZE = 10
    sample_bias = 1.05                                        # probability weighting of [sample_bias ** i for i in range(1,buffer_size-batch_size)]
    cash_bias = -9999
    if mode == "train":
        LRA = 9e-5                                            # Learning rate for Actor (training)
    else:
        LRA = 1e-5                                            # Learning rate for Actor (testing)
    train_rolling_steps = 1
    test_rolling_steps = 0
    window_size = 20                                          # window size per input
    tickers_num = len(tickers)                                # the number of assets (exclude Cash)
    action_size = tickers_num + 1
    feature_num = 11                                          # number of features (adjusted open, close, high, low)
    state_dim = (tickers_num, window_size, feature_num)       # number of assets x window size x number of features
    epsiode_reward = []                                       # reward of each episode = Cumulative Return by CNN / Cumulative Return by UCR
    np.random.seed(1337)
    total_step = 0

    # Tensorflow GPU optimization
    sess = gpu_settings()

    # create network and replay buffer
    actor = ActorNetwork(sess, state_dim, action_size, BATCH_SIZE, LRA)
    buff = ReplayBuffer(BUFFER_SIZE, sample_bias)

    # start simulation
    print("Portfolio Management Simualation Experiment Start ({})".format(mode))
    print("{} period: {} to {}".format(mode,start,end))

    # iterate the episode
    for ep in range(ep_start+1, ep_start+ep_train+1):

        if not (mode == "train" and ep != ep_start+1):
            load_weights(mode, actor, ep)

        train_tickers = prepare_tickers(train_mode, mode, tickers, sector, start, end)

        # construct a portfolio within defined period
        P = Portfolio(train_tickers, start, end, mode=mode)
        if not P.consistent:
            exit(0)

        # construct states
        state = prepare_states(P, window_size)

        cumulated_return = 1

        # iterate the defined period
        for step in range(len(state)-2):

            # print(state[step])

            # extract the last action
            if step == 0:
                last_action = np.array([0 for _ in range(state_dim[0])])
            else:
                last_action = np.array(P.portfolio_weights[-1][:state_dim[0]])

            # prediction of tomorrow price
            prediction = np.array([1 for _ in range(state_dim[0])])


            # generate action, single batch = batch only conisists one element
            action = actor.model.predict([state[step].reshape([1, state_dim[2], state_dim[1], state_dim[0]]),
                                          last_action.reshape(1, state_dim[0]), np.array([[cash_bias]]), prediction.reshape(1, state_dim[0])])


            # generate daily return
            day_return, future_price = P.calculate_return(action[0], last_action, step)


            # extract historical data to re-train the model at each time step
            experience_replay(actor, buff, BATCH_SIZE, mode,  state, step, future_price, last_action, cash_bias, prediction,
                              train_rolling_steps, test_rolling_steps, sess)


            cumulated_return = cumulated_return *  day_return
            total_step += 1
            print("Episode {} Step {} Date {} Cumulated Return {} Day return {}".format(ep, total_step,
                                                                                        P.df_normalized.index[P.start_index+step+1].strftime('%Y-%m-%d'),
                                                                                        cumulated_return, day_return))
            print(action[0])


        # No trading at last day, clear the portfolio and set return to 1
        P.portfolio_weights.append(np.array([0 for _ in range(tickers_num)] + [1]))

        # Generate Uniform Constant Rebalanced Portfolios strategy
        P.UCRP()
        P.UBHP()

        # calculate performance: how cumulative return outperforms Uniform Constant Rebalanced Portfolios strategy
        performance = cumulated_return / np.array(P.portfolio_UCRP).cumprod()[-1]
        epsiode_reward.append(performance)

        # save the model and result
        save_weights_results(P, mode, actor, ep, performance)

        # clear the Portfolio and buffer
        P.clear()
        buff.erase()
        if mode == 'valid':
            total_step = 0


        print("")


    # save_epsoide_performance(mode, start, end, epsiode_reward)


    print("Finish")
    print("")

    return epsiode_reward

def train(ep_start=0, ep_end=30):
    Simulate('2010-01-04', '2015-12-31', mode="train", ep_train=ep_end-ep_start, ep_start=ep_start)

def valid(ep_start = -1, ep_end = -1):
    if ep_end == -1:
        Simulate('2016-01-04', '2017-12-31', mode="valid", ep_train=1 , ep_start=ep_start)

    # validate performance of each episode
    else:
        print(Simulate('2016-01-04', '2017-12-31', mode="valid", ep_train=ep_end-ep_start, ep_start=ep_start+1))

def test(ep_start=30):
    Simulate('2016-01-04', '2017-12-31', mode="test",  ep_train=1 ,  ep_start=ep_start)


if __name__ == "__main__":
    # train period: '2010-01-04' to '2015-12-31'
    # valid period: '2015-01-02' to '2015-12-31'
    # test period:  '2016-01-04' to '2017-12-31'

    # train(ep_start=303, ep_end=400)
    valid(ep_start=0, ep_end=400)
    # test(ep_start=40)