import numpy as np
import math
from keras.models import Model
from keras.layers import Convolution2D, Flatten, Input, concatenate, Reshape, Dense, Permute, LSTM, TimeDistributed, multiply
from keras.layers.core import Activation
import tensorflow as tf
import keras.backend as K
from keras import regularizers


class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.state_size = state_size

        # Global step for decay learning rate
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.LEARNING_RATE = tf.train.exponential_decay(LEARNING_RATE, self.global_step,
                                                        50000, 0.1, staircase=False)

        # Sets the global TensorFlow session
        K.set_session(sess)

        # Now create the model
        self.model , self.weights, self.state, self.last_action, self.cash_bias, self.prediction, self.test = self.CNN_Dense(state_size)

        # future price includes cash bias which is 1 at all
        self.future_price = tf.placeholder(tf.float32, [None, state_size[0]+1])

        # Reward function to be maximized
        self.reward = self.reward_3()

        # perform gradient ascending
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.reward, global_step=self.global_step)
        self.sess.run(tf.global_variables_initializer())

    def reward_1(self):
        # Average logarithmic cumulated return
        print("Reward function 1 (Average logarithmic cumulated return)")

        # transaction cost
        self.transaction_factor = 0.001
        self.transaction_cost = 1 - tf.reduce_sum(self.transaction_factor * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)

        return -tf.reduce_mean(tf.log(self.transaction_cost * tf.reduce_sum(self.model.output * self.future_price, axis=1)))

    def reward_2(self):
        # Average Uniform Constant Rebalanced reward
        print("Reward function 2 (Average Uniform Constant Rebalanced reward)")

        # transaction cost
        self.transaction_factor = 0.001
        self.transaction_cost = 1 - tf.reduce_sum(self.transaction_factor * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)

        return -tf.reduce_mean(tf.log(self.transaction_cost * tf.reduce_sum(self.model.output * self.future_price, axis=1) /
                                      tf.reduce_sum(self.future_price[:,:-1] / self.state_size[0], axis=1)))

    def reward_3(self):
        # Sharpe ratio
        print("Reward function 3 (Sharpe ratio)")

        # transaction cost
        self.transaction_factor = 0.001
        self.transaction_cost = 1 - tf.reduce_sum(self.transaction_factor * tf.abs(self.model.output[:,:-1] - self.last_action), axis=1)

        # Daily average portfolio return
        self.average_return = tf.log(self.transaction_cost * tf.reduce_sum(self.model.output * self.future_price, axis=1))

        # Risk free rate for sharpe ratio (Three-month U.S. Treasury bill)
        self.risk_free_rate = np.log(1.0148) / 60

        return -(tf.reduce_mean(self.average_return)-self.risk_free_rate) * \
               tf.sqrt(tf.to_float(tf.size(self.average_return))) / K.std(self.average_return)

    def train(self, states, last_actions, future_prices, cash_bias, prediction):
        # train model in batch to optimize reward
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.last_action: last_actions,
            self.future_price: future_prices,
            self.cash_bias: cash_bias,
            self.prediction: prediction
        })

    def CNN(self, state_size):
        print("Now we build actor (CNN only)")

        # Network Hyperparameters
        Conv2D_1_kernel = (5,1)
        Conv2D_1_filters = 3
        Conv2D_2_kernel = (state_size[1]-Conv2D_1_kernel[0]+1, 1)
        Conv2D_2_filters = 20

        # input layer
        State = Input(shape=[state_size[2], state_size[1], state_size[0]])
        prediction = Input(shape=[state_size[0]])

        # last action excludes cash bias component
        last_action = Input(shape=[state_size[0]])
        last_action_1 = Reshape((1, 1, state_size[0]))(last_action)

        # cash bias as constant input
        cash_bias = Input(shape=[1])

        # a set of convolution layers designed to extract 3 days pattern
        Conv2D_1 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, state_size[2], state_size[1], state_size[0]),   # state_size[1] = time interval, state_size[0] = number of assests
            filters=Conv2D_1_filters,
            kernel_size=Conv2D_1_kernel,      # filters: make sure the row size of kernel is one -> independent assets
            strides=1,
            padding='valid',                  # Padding method
            data_format='channels_first',     # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(State)

        Conv2D_2 = Convolution2D(
            batch_input_shape= (self.BATCH_SIZE, Conv2D_1_filters, state_size[1]-Conv2D_1_kernel[0]+1, state_size[0]),
            filters=Conv2D_2_filters,
            kernel_size=Conv2D_2_kernel,      # filters: make sure the row size of kernel is one -> independent assets
            strides=1,
            padding='valid',                   # Padding method
            data_format='channels_first',      # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer= regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(Conv2D_1)

        Concate = concatenate([Conv2D_2, last_action_1], axis=1)

        Conv2D_3 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, Conv2D_2_filters+1, 1, state_size[0]),
            filters=1,
            kernel_size=(1, 1),               # filters: make sure the row size of kernel is one -> independent assets
            strides=1,
            padding='valid',                  # Padding method
            data_format='channels_first',     # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
        )(Concate)

        # Flatten
        vote = Flatten()(Conv2D_3)

        # integrate with prediction
        vote_p = multiply([vote, prediction])

        F1 = concatenate([vote_p,cash_bias], axis=1)

        # output layer (actions)
        action = Activation('softmax')(F1)

        model = Model(inputs=[State, last_action, cash_bias, prediction], outputs=action)
        return model, model.trainable_weights, State, last_action, cash_bias, prediction, F1

    def CNN_Dense(self, state_size):
        print("Now we build network (CNN + Dense)")

        # Network Hyperparameters
        Conv2D_1_kernel = (5,1)
        Conv2D_1_filters = 3
        Conv2D_2_kernel = (5,1)
        Conv2D_2_filters = 20
        Dense_units = 100

        # input layer
        State = Input(shape=[state_size[2], state_size[1], state_size[0]])
        prediction = Input(shape=[state_size[0]])

        # last action excludes cash bias component
        last_action = Input(shape=[state_size[0]])
        last_action_1 = Reshape((state_size[0], 1))(last_action)

        # cash bias as constant input
        cash_bias = Input(shape=[1])

        # a set of convolution layers designed to extract 3 days pattern
        Conv2D_1 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, state_size[2], state_size[1], state_size[0]),   # state_size[1] = time interval, state_size[0] = number of assests
            filters=Conv2D_1_filters,
            kernel_size=Conv2D_1_kernel,      # filters: make sure the row size of kernel is one -> independent assets
            strides=1,
            padding='valid',                  # Padding method
            data_format='channels_first',     # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(State)

        Conv2D_2 = Convolution2D(
            batch_input_shape= (self.BATCH_SIZE, Conv2D_1_filters, state_size[1]-Conv2D_1_kernel[0]+1, state_size[0]),
            filters=Conv2D_2_filters,
            kernel_size=Conv2D_2_kernel,       # filters: make sure the row size of kernel is one -> independent assets
            strides= 1,
            padding='valid',                   # Padding method
            data_format='channels_first',      # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer= regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(Conv2D_1)

        # Pass the feature maps according to assets one by one
        L1 = Permute((3,1,2))(Conv2D_2)
        L2 = Reshape((state_size[0], -1))(L1)
        L3 = concatenate([last_action_1, L2], axis=2)
        L4 = TimeDistributed(Dense(Dense_units, activation='relu'), input_shape=[state_size[0],-1])(L3)
        L5 = TimeDistributed(Dense(1, activation='linear'), input_shape=[state_size[0], Dense_units])(L4)

        # Generate voting score of each asset
        vote = Flatten()(L5)

        # integrate with prediction
        vote_p = multiply([vote, prediction])

        F1 = concatenate([vote_p,cash_bias],axis=1)

        # output layer (actions)
        action = Activation('softmax')(F1)

        model = Model(inputs=[State, last_action, cash_bias, prediction], outputs=action)
        return model, model.trainable_weights, State, last_action, cash_bias, prediction, F1

    def CNN_LSTM(self, state_size):
        print("Now we build network (CNN + LSTM)")

        # Network Hyperparameters
        Conv2D_1_kernel = (5,1)
        Conv2D_1_filters = 3
        Conv2D_2_kernel = (5,1)
        Conv2D_2_filters = 20
        Lstm_units = 10

        # input layer
        State = Input(shape=[state_size[2], state_size[1], state_size[0]])
        prediction = Input(shape=[state_size[0]])

        # last action excludes cash bias component
        last_action = Input(shape=[state_size[0]])

        # cash bias as constant input
        cash_bias = Input(shape=[1])

        # a set of convolution layers designed to extract 3 days pattern
        Conv2D_1 = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, state_size[2], state_size[1], state_size[0]),   # state_size[1] = time interval, state_size[0] = number of assests
            filters=Conv2D_1_filters,
            kernel_size=Conv2D_1_kernel,      # filters: make sure the row size of kernel is one -> independent assets
            strides=1,
            padding='valid',                  # Padding method
            data_format='channels_first',     # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(State)

        Conv2D_2 = Convolution2D(
            batch_input_shape= (self.BATCH_SIZE, Conv2D_1_filters, state_size[1]-Conv2D_1_kernel[0]+1, state_size[0]),
            filters=Conv2D_2_filters,
            kernel_size=Conv2D_2_kernel,       # filters: make sure the row size of kernel is one -> independent assets
            strides=1,
            padding='valid',                   # Padding method
            data_format='channels_first',      # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer= regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
            activation='relu'
        )(Conv2D_1)

        # Pass the feature maps according to assets one by one
        L1 = Permute((3,2,1))(Conv2D_2)
        L2 = TimeDistributed(LSTM(units=Lstm_units, activation='tanh', return_sequences=True), input_shape=[state_size[0],-1,Conv2D_2_filters])(L1)
        L3 = TimeDistributed(LSTM(units=1, activation='linear', return_sequences=False), input_shape=[state_size[0],-1,Lstm_units])(L2)

        # Generate voting score of each asset
        vote = Flatten()(L3)

        # integrate with prediction
        vote_p = multiply([vote, prediction])

        F1 = concatenate([vote_p,cash_bias],axis=1)

        # output layer (actions)
        action = Activation('softmax')(F1)

        model = Model(inputs=[State, last_action, cash_bias, prediction], outputs=action)
        return model, model.trainable_weights, State, last_action, cash_bias, prediction, vote

    def LSTM(self, state_size):
        print("Now we build network (LSTM only)")

        # Network Hyperparameters
        LSTM_units = 100

        # input layer
        State = Input(shape=[state_size[2], state_size[1], state_size[0]])
        prediction = Input(shape=[state_size[0]])

        # last action excludes cash bias component
        last_action = Input(shape=[state_size[0]])

        # cash bias as constant input
        cash_bias = Input(shape=[1])

        # LSTM layers
        L1 = Permute((3,2,1))(State)
        L2 = TimeDistributed(LSTM(units=LSTM_units, activation='tanh', return_sequences=True), input_shape=[state_size[0],state_size[1],state_size[2]])(L1)
        L3 = TimeDistributed(LSTM(units=1, activation='linear', return_sequences=False), input_shape=[state_size[0],state_size[1],LSTM_units])(L2)

        # Flatten
        vote = Flatten()(L3)

        # integrate with prediction
        vote_p = multiply([vote, prediction])

        F1 = concatenate([vote_p, cash_bias],axis=1)

        # output layer (actions)
        action = Activation('softmax')(F1)

        model = Model(inputs=[State, last_action, cash_bias, prediction], outputs=action)
        return model, model.trainable_weights, State, last_action, cash_bias, prediction, F1

    def LSTM_CNN(self, state_size):
        print("Now we build network (LSTM + CNN)")

        # Network Hyperparameters
        LSTM_units = 30

        # input layer
        State = Input(shape=[state_size[2], state_size[1], state_size[0]])
        prediction = Input(shape=[state_size[0]])

        # last action excludes cash bias component
        last_action = Input(shape=[state_size[0]])
        last_action_1 = Reshape((1, 1, state_size[0]))(last_action)

        # cash bias as constant input
        cash_bias = Input(shape=[1])

        # LSTM layers
        L1 = Permute((3,2,1))(State)
        L2 = TimeDistributed(LSTM(units=LSTM_units, activation='tanh', return_sequences=False), input_shape=[state_size[0],state_size[1],state_size[2]])(L1)
        L3 = Permute((2,1))(L2)
        L4 = Reshape((LSTM_units,1,state_size[0]))(L3)

        Concate = concatenate([L4, last_action_1], axis=1)

        Conv2D = Convolution2D(
            batch_input_shape=(self.BATCH_SIZE, LSTM_units+1, 1, state_size[0]),
            filters=1,
            kernel_size= (1, 1),              # filters: make sure the row size of kernel is one -> independent assets
            strides=1,
            padding='valid',                  # Padding method
            data_format='channels_first',     # (batch_size, channels, rows, cols),
            kernel_regularizer= regularizers.l2(1e-8),
            activity_regularizer=regularizers.l2(1e-8),
            bias_regularizer = regularizers.l2(1e-8),
        )(Concate)

        # Flatten
        vote = Flatten()(Conv2D)

        # integrate with prediction
        vote_p = multiply([vote, prediction])

        F1 = concatenate([vote_p, cash_bias],axis=1)

        # output layer (actions)
        action = Activation('softmax')(F1)

        model = Model(inputs=[State, last_action, cash_bias, prediction], outputs=action)
        return model, model.trainable_weights, State, last_action, cash_bias, prediction, F1