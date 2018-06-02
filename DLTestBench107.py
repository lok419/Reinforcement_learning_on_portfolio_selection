import numpy
import pandas
import os
import pickle
import datetime
import math
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

input_timesteps = 120
predict_timesteps = 5
data_dim = 6
num_portfolio = 5

test_name = "model107"

filepath = "./time"
#filepath = "/Users/Vanguardcreed/Desktop/time"
destinationpath = "./result"
#destinationpath = "/Users/Vanguardcreed/Desktop/time"

start_date = datetime.datetime.strptime( '1/1/10', "%m/%d/%y")
split_date = datetime.datetime.strptime( '1/1/16', "%m/%d/%y")

def get_dates(mode="all"):
    # return an array containing the dates
    df = pandas.read_csv("{}/date.csv".format(filepath))
    df['Date'] = pandas.to_datetime(df['Date'])
    if mode=="train":
        result = df['Date'][(df['Date']<split_date) & (df['Date']>start_date)]
    elif mode == "test":
        train = df.loc[(df['Date']<split_date) & (df['Date']>start_date)].index
        limit = train[-1]
        result = df['Date'].iloc[limit-input_timesteps+1:]
    else:
        return
    return result

def get_model(LSTM_units=200, dropout=0.2):

    # return the model
    def conv_unit(x, depth, field, strides=1, padding="same"):
        x = Conv1D(depth,field, strides = strides, padding=padding)(x)
        x = BatchNormalization(axis=1,momentum=0.99)(x)
        x = Activation('relu')(x)
        return x

    input_tensor = Input(shape=(input_timesteps, data_dim,))
    x = conv_unit(input_tensor, 32, 5, strides=2, padding="valid")

    branch11 = conv_unit(x, 64, 1)
    branch31 = conv_unit(x, 64, 3)
    branch51 = conv_unit(x, 64, 5)
    branch_pool1 = AveragePooling1D(3, strides=1, padding='same')(x)

    x = concatenate([branch11, branch31, branch51, branch_pool1])

    branch32 = conv_unit(x, 128, 3)
    branch52 = conv_unit(x, 128, 5)
    branch72 = conv_unit(x, 128, 7)
    branch_pool2 = AveragePooling1D(3, strides=1, padding="same")(x)

    x = concatenate([branch32, branch52, branch72, branch_pool2])

    branch33 = conv_unit(x, 192, 3)
    branch73 = conv_unit(x, 192, 7)
    branch103 = conv_unit(x, 192, 10)
    branch_pool3 = conv_unit(x, 128,1)
    branch_pool3 = AveragePooling1D(3, strides=1, padding='same')(branch_pool3)
    x = concatenate( [branch33, branch73, branch103, branch_pool3])
    x = Dropout(dropout)(x)
    x = LSTM(LSTM_units, activation="tanh", recurrent_activation="hard_sigmoid",return_sequences=True)(x)
    x = Dropout(dropout)(x)
    x = LSTM(LSTM_units,activation="tanh", recurrent_activation="hard_sigmoid", return_sequences=False)(x)
    x = Dropout(dropout)(x)
    x = Dense(100, activation="tanh")(x)
    x = Dense(1, activation="tanh")(x)
    model = Model(input_tensor, x)
    optimizer = Adam(lr=0.0001, beta_1=0.8, beta_2=0.9)
    model.compile(loss='mse', optimizer='adam')
    return model

def load_test(filepath):
    model = load_model(filepath)
    test_model(model)
    return

def check_y(y):
    print("check y")
    base = len(y)
    extreme = [i for i in y if i>=1 or i<=-1]
    print("extreme case is" + str(len(extreme)/base))
    positive = sum( [1 for i in y if i>0])/base
    print("positive case is" + str(positive))
    returnx

def train_model(model, num_epochs=100, batch_size=256, start_from=0):

    if os.path.exists("./train_features.pickle"):
        with open('./train_features.pickle', 'rb') as file:
            X,y = pickle.load(file)
            print("loaded features and label")
    else:
        # list of trained or tested date
        dates = get_dates(mode = "train")

        # list of daily data
        dflist = []
        new_dates = []

        # fetch the daily data
        for date in dates:
            try:
                df = pandas.read_csv("{}/{}.csv".format(filepath, date.strftime("%Y-%m-%d")))
                dflist.append(df)
                new_dates.append(date)
                print("read {}.csv".format(date.strftime("%Y-%m-%d")))
            except:
                print("cannot read {}/{}.csv".format(filepath, date.strftime("%Y-%m-%d")))

        dates = new_dates
        data_len = input_timesteps + predict_timesteps

        # prepare the first set of features and labels
        X,y = assemble_data(dflist[0:data_len])

        for i in range(1, len(dflist)-data_len+1, 5):
            tmp_X, tmp_y = assemble_data(dflist[i:i+data_len])

            X = numpy.concatenate((X,tmp_X), axis=0)
            y = numpy.concatenate((y,tmp_y), axis=0)
            print("Prepared feature set of of {}".format(dates[i].strftime("%Y-%m-%d")))

        with open('train_features.pickle', 'wb') as file:
            pickle.dump([X,y],file)

    if start_from != 0:
        try:
            model.load_weights("{}/model_%03d.hdf5".format(destinationpath)%start_from)
            print("loaded weights ({})".format(start_from))
        except:
            print("unable to load weights ({})".format(start_from))
            print("start from 0 epoch")

    print("training period: {} to {}".format(start_date.strftime("%Y-%m-%d"),split_date.strftime("%Y-%m-%d")))
    print("input timesetpes: {} day".format(input_timesteps))
    print("predict timesteps: {} day".format(predict_timesteps))

    checkpoint = ModelCheckpoint(destinationpath+"/model2_{epoch:03d}.hdf5", verbose=1, save_best_only=False)
    history = model.fit(X, y, epochs=num_epochs-start_from, batch_size=batch_size, shuffle=False, callbacks=[checkpoint])
    trainingDoc = pandas.DataFrame()
    trainingDoc['loss'] = history.history['loss']
    trainingDoc.to_csv("{}/train_loss_{}.csv".format(destinationpath, test_name))
    print("saved train loss of {}".format(test_name))
    return model

def assemble_data(dates_df):
    # preprocess the data and construct a input matrix X and label Y to network

    # number of stocks to be trained
    num_samples = len(dates_df[0].index)
    X = []
    y = []

    for stock in range(num_samples):
        # for each stock:
        SX = []
        for i in range(0, input_timesteps):
            # 8 financial indicators: Open,High,Low,Close,Adj_close,Volume,Scaled_adj_close,Scaled_volume
            SX.append(dates_df[i].iloc[stock, 2:10].values)
        SX = numpy.array(SX)

        # close price as label
        sy = dates_df[input_timesteps+predict_timesteps-1].iloc[stock, 5]

        # logarithm change of close price
        tmp_y = (sy /SX[input_timesteps-1][3])
        y.append(math.log(tmp_y))

        # preprocessing
        scaler = MinMaxScaler(feature_range=(0.1,0.9))

        # dropped adjusted close and volume, scale the Open,High,Low,Close
        tmp_x = numpy.concatenate( (SX[:,6:8],scaler.fit_transform(SX[:, 0:4]) ),axis=1)

        # add to main features
        X.append(tmp_x)

        # y.append( (tmp_y[0][3] - tmp_x[input_timesteps-1][5])/tmp_x[input_timesteps-1][5] )
        # tmp = numpy.concatenate( (sy[6:8].reshape(1,-1),scaler.transform(sy[0:4].reshape(1,-1))), axis=0)
    return numpy.array(X), numpy.array(y).reshape(-1,1)

def test_model(model):

    def top_k(data):
        top_list = [0] * num_portfolio
        position_list = [-1]* num_portfolio
        for i in range(len(data)):
            if data[i] > top_list[0]:
                top_list, position_list = insert_top_k(data[i],i,top_list,position_list)
        return top_list, position_list

    #auxiliary fucntion for top_k
    def insert_top_k(sim, pos, top_list, position_list):
        limit = len(top_list)
        for i in range(1, limit + 1):
            if i == limit:
                top_list[i-1] = sim
                position_list[i-1] = pos
            elif top_list[i] < sim:
                top_list[i-1] = top_list[i]
                position_list[i-1] = position_list[i]
            else:
                top_list[i-1] = sim
                position_list[i-1] = pos
                break
        return top_list, position_list

    dates = get_dates(mode="test")
    column_list = ['Date','MSE','Accuracy'] + [str(i) for i in range(0, num_portfolio)]
    result_df = pandas.DataFrame(columns=column_list)
    dflist = []
    new_dates = []
    for date in dates:
        try:
            #print(filepath+"/"+dat.strftime("%Y-%m-%d")+".csv")
            df = pandas.read_csv("{}/{}.csv".format(filepath,date.strftime("%Y-%m-%d")))
            dflist.append(df)
            new_dates.append(date)
        except:
            pass
    data_len = input_timesteps + predict_timesteps
    test_date = new_dates[input_timesteps-1: len(new_dates)-predict_timesteps]
    for i in range(0, len(test_date)):
        print(test_date[i])
        X_test, y_test = assemble_data(dflist[i:i+data_len])
        y_pred = model.predict(X_test)[:,0]
        print(y_pred)
        print(y_test)
        #mse = (( y_pred - y_test ) ** 2).mean(axis=None)
        mse = ((y_pred-y_test)**2).mean()
        y_act = X_test[:, input_timesteps-1 ,5]
        change = (y_pred - y_act)
        pred, portfolio = top_k(change)
        direction = (y_test[:,0] - y_act)
        accuracy = sum([1 for x,y in zip(y_pred,y_test) if x*y>0] )/direction.shape[0]
        print(accuracy)
        result_df = result_df.append(pandas.Series([test_date[i], mse, accuracy]+portfolio,index=column_list), ignore_index=True)
        #result_df[i] = numpy.nparray([test_date[i], mse, accuracy] + portfolio)

    result_df.to_csv("{}/{}_test.csv".format(destinationpath,test_name))
    return

def run():
    if os.path.exists("{}/{}.h5".format(destinationpath,test_name)):
        model = load_model("{}/{}.h5".format(destinationpath,test_name))
        print("loaded model")
    else:
        model = get_model()
        model = train_model(model, start_from=81)
        try:
            model.save("{}/{}.h5".format(destinationpath,test_name))
            print("model {}.h5 is saved".format(test_name))
        except:
            print("Error: model is not saved")
    test_model(model)
    return

run()
