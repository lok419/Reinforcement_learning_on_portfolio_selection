import numpy
import pandas
import os
import datetime
import math
from keras.layers import *
from keras.models import *
from keras.metrics import *
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

input_timesteps = 180
predict_timesteps = 20
data_dim = 6
num_portfolio = 5

test_name = "model107"

filepath = "/home/u3514106/time"
#filepath = "/Users/Vanguardcreed/Desktop/time"
destinationpath = "/home/u3514106/result"
#destinationpath = "/Users/Vanguardcreed/Desktop/time"

start_date = datetime.datetime.strptime( '1/1/10', "%m/%d/%y")
split_date = datetime.datetime.strptime( '1/1/14', "%m/%d/%y")

def get_dates(mode="all"):
    df = pandas.read_csv("date.csv")
    df['Date'] = pandas.to_datetime(df['Date'])
    if mode=="train":
        result = df['Date'][(df['Date']<split_date) & (df['Date']>start_date)]
    elif mode == "test":
        train = df.loc[(df['Date']<split_date) & (df['Date']>start_date)].index
        limit = train[len(train)-1]
        result = df.iloc[1, limit-input_timesteps+1:]
    else:
        pass
    return result

def get_model(LSTM_units=200, dropout=0.2):
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
    return

def train_model(model, num_epochs=100, batch_size=256):
    dates = get_dates(mode = "train")
    dflist = []
    new_dates = []
    for dat in dates:
        try:
            #print(filepath+"/"+dat.strftime("%Y-%m-%d")+".csv")
            df = pandas.read_csv(filepath+"/"+dat.strftime("%Y-%m-%d")+".csv")
            dflist.append(df)
            new_dates.append(dat)
        except:
            pass
    dates = new_dates
    data_len = input_timesteps + predict_timesteps
    X,y = assemble_data(dflist[0:data_len])
    for i in range(1, len(dflist)-data_len+1, 5):
        print(dates[i])
        tmp_X, tmp_y = assemble_data(dflist[i:i+data_len])
        X = numpy.concatenate((X,tmp_X), axis=0)
        y = numpy.concatenate((y,tmp_y), axis=0)
    history = model.fit(X, y, epochs=num_epochs, batch_size=batch_size, shuffle=False)
    trainingDoc = pandas.DataFrame()
    trainingDoc['loss'] = history.history['loss']
    trainingDoc.to_csv(destinationpath+ "/train_loss_" + test_name + ".csv")
    return model

def assemble_data(dates_df):
    num_samples = len(dates_df[0].index)
    X = []
    y = []
    for stock in range(num_samples):
        SX = []
        for i in range(0, input_timesteps):
            SX.append(dates_df[i].iloc[stock, 2:10].values)
        SX = numpy.array(SX)
        sy = dates_df[input_timesteps+predict_timesteps-1].iloc[stock, 5]
        tmp_y = (sy /SX[input_timesteps-1][3])
        y.append(math.log(tmp_y))
        #preprocessing
        scaler = MinMaxScaler(feature_range=(0.1,0.9))
        tmp_x = numpy.concatenate( (SX[:,6:8],scaler.fit_transform(SX[:, 0:4]) ),axis=1)
        X.append(tmp_x)
        #y.append( (tmp_y[0][3] - tmp_x[input_timesteps-1][5])/tmp_x[input_timesteps-1][5] )
        #tmp = numpy.concatenate( (sy[6:8].reshape(1,-1),scaler.transform(sy[0:4].reshape(1,-1))), axis=0)
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
    for dat in dates:
        try:
            #print(filepath+"/"+dat.strftime("%Y-%m-%d")+".csv")
            df = pandas.read_csv(filepath+"/"+dat.strftime("%Y-%m-%d")+".csv")
            dflist.append(df)
            new_dates.append(dat)
        except:
            pass
    data_len = input_timesteps + predict_timesteps
    test_date = new_dates[input_timesteps-1: len(new_dates)-predict_timesteps]
    for i in range(0, len(test_date)):
        print(test_date[i])
        X_test, y_test = assemble_data(dflist[i:i+data_len])
        y_pred = model.predict(X_test)[:,0]
        #mse = (( y_pred - y_test ) ** 2).mean(axis=None)
        mse = ((y_pred-y_test)**2).mean()
        y_act = X_test[:, input_timesteps-1 ,5]
        change = (y_pred - y_act)
        pred, portfolio = top_k(change)
        direction = (y_test[:,0] - y_act)
        accuracy = sum([1 for x,y in zip(change,direction) if x*y>0] )/direction.shape[0]
        result_df = result_df.append(pandas.Series([test_date[i], mse, accuracy]+portfolio,index=column_list), ignore_index=True)
        #result_df[i] = numpy.nparray([test_date[i], mse, accuracy] + portfolio)

    result_df.to_csv(destinationpath+"/"+test_name+"_test.csv")
    return

def run():
    model = get_model()
    model = train_model(model)
    try:
        model.save(destinationpath+"/"+ test_name +".h5")
        print("model is saved")
    except:
        print("model is not saved")
    test_model(model)
    return

run()
