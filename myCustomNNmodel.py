from keras.models import Sequential as Seq
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler as ss

class my_RNN_LSTM_Regressor():

    def __init__(self, x_lstm, x_ann, y, lstm_time_steps):

        # LSTM Model
        model_lstm = Seq()
        model_lstm.add(LSTM(units=50, activation='relu', input_shape=(lstm_time_steps, 1)))
        model_lstm.add(Dense(units=1))

        # ANN Model
        model_ann = Seq()
        model_ann.add(Dense(units=50, activation='relu', input_shape=(x_ann.shape[1],)))
        model_ann.add(Dense(units=1))

        combined_input = concatenate([model_lstm.output, model_ann.output])

        # LAST Layer
        x = Dense(32, activation="relu")(combined_input)
        x = Dense(1)(x)
        
        # Compiling the Model
        model = Model(inputs=[model_lstm.inputs, model_ann.inputs], output=x)

        # Split model -> Train/Test
        self.train_test_split(self, df, test_num)
        self.time_steps_split(self, time_steps)
        # Evaluate model



def train_test_split(self, x_lstm, x_ann, y, test_num):