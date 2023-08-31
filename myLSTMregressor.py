from keras.models import Sequential as sq
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as ss

class my_RNN_LSTM_Regressor():

    def __init__(self, x, y, test_num, time_steps, scaler, units=50, dropout=0.2, epoch=50, batch_size=32, predict=False, product=''):
        
        self.pred_arg_1 = y.columns[0]
        self.pred_arg_2 = time_steps
        self.pred_arg_3 = scaler
        self.pred_arg_4 = product
        
        df = x

        print('Processing: Train-Test Split...')
        self.train_set, self.test_set = self.train_test_split(df, test_num)
        
        print('Processing: Seperate Train Data to X and Y according to Time Steps...')
        self.X_train, self.y_train = self.time_steps_split(time_steps, y.columns[0])

        print('Model is building...')
        target_len = y.shape[1]
        self.build(units, dropout, target_len)

        print('Model is running...')
        self.run(epoch, batch_size, x.shape[1])



        if predict:
            print('Estimating by model...')
            self.predict(self.pred_arg_1, self.pred_arg_2, self.pred_arg_3, self.pred_arg_4)
        
    def train_test_split(self, df, test_num):
        train_set = df.iloc[test_num:, :].reset_index(drop=True).sort_index(ascending = False)
        test_set = df.iloc[:test_num, :].reset_index(drop=True).sort_index(ascending = False)
        # train and test set sorted old to new!

        return train_set, test_set

    def time_steps_split(self, time_steps, target_col):

        X_train = []
        y_train = []
        for i in range(time_steps, self.train_set.shape[0]):
            X_train.append(self.train_set.iloc[i-time_steps:i,:].values)
            # X_train shape: time_steps x x_col_len
            y_train.append(self.train_set.loc[i, [target_col]].values)

        X_train, y_train = np.array(X_train), np.array(y_train)
        # X_train shape = (train_set - time_steps) x time_steps x x_col_len
        return X_train, y_train

            
    def build(self, units, dropout, target_len):
        self.model = sq()

        # input
        self.model.add(LSTM(units=units, return_sequences = True, input_shape = (self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(dropout))
        
        # first
        self.model.add(LSTM(units = units, return_sequences = True))
        self.model.add(Dropout(dropout))

        # second
        self.model.add(LSTM(units = units, return_sequences = True))
        self.model.add(Dropout(dropout))

        # thirth
        self.model.add(LSTM(units = units, return_sequences = False))
        self.model.add(Dropout(dropout))
        
        # output 
        self.model.add(Dense(units=target_len))


    def run(self, epoch, batch_size, x_col_len):
        # compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # fit model

        # no problem until here in case of shape of X and y train!
        self.model.fit(self.X_train, self.y_train, epochs = epoch, batch_size = batch_size)

    def predict(self, target_col, time_steps, scaler, product):

        import matplotlib.pyplot as plt

        dataset_X_total = pd.concat((self.train_set[:], self.test_set[:]), axis = 0)

        inputs = dataset_X_total.iloc[len(dataset_X_total) - len(self.test_set) - time_steps:].values # last (time_steps) data in train set + test set

        X_test = []
        for i in range(time_steps, time_steps + len(self.test_set)):
            X_test.append(inputs[i-time_steps:i, :])

        X_test = np.array(X_test)
        # X_test shape: len(self.test_set) x time_steps x x_col_len

        predicted_prices = self.model.predict(X_test)

        predicted_prices_inv = scaler.inverse_transform(predicted_prices)
        test_set_target_inv = scaler.inverse_transform(self.test_set[target_col].values.reshape(-1,1))
    
        plt.plot(test_set_target_inv, color='blue', label = f'{product} Real Price')
        plt.plot(predicted_prices_inv, color='tomato', label = f'{product} Estimated Price')
        plt.title(f'{product} Market Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        print('\n', 'Process is succeeded! Model results are in shown!')