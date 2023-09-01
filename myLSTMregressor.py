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
        self.run(epoch, batch_size)

        if predict:
            print('Estimating by model...')
            self.predict(self.pred_arg_1, self.pred_arg_2, self.pred_arg_3, self.pred_arg_4)
        

    def train_test_split(self, df, test_num):
        train_set = df.iloc[test_num:, :].reset_index(drop=True).sort_index(ascending = False).reset_index(drop=True)
        test_set = df.iloc[:test_num, :].reset_index(drop=True).sort_index(ascending = False).reset_index(drop=True)
        # train and test set sorted old to new!
        print('TRAIN_SET ', train_set.shape)
        print(f'My expectation: {len(df)-test_num} x {len(df.columns)}')
        print('-'*20)
        print('TEST_SET ', test_set.shape)
        print(f'My expectation: {test_num} x {len(df.columns)}')
        print('-'*20)
        return train_set, test_set

    def time_steps_split(self, time_steps, target_col):
        X_train = []
        y_train = []
        train_set_without_target = self.train_set.drop([target_col], axis = 1)
        for i in range(time_steps, self.train_set.shape[0]):
            time_steps_arr = self.train_set.loc[i-time_steps:i-1,[target_col]].values
            if i == time_steps:
                print('time_steps_arr ', time_steps_arr.shape)
                print(f'My expectation: 1 x {time_steps}')
                print('-'*20)
            # time_steps x 1  
            features_arr = train_set_without_target.iloc[i-1:i, :].values.reshape(-1, 1)
            if i == time_steps:
                print('features_arr ', features_arr.shape)
                print('My expectation: 1 x 15')
                print('-'*20)

            # (train_set_len - 1) x 1
            X_train.append(np.concatenate((time_steps_arr, features_arr), axis = 0))

            y_train.append(self.train_set.loc[i, [target_col]].values)

        X_train, y_train = np.array(X_train), np.array(y_train)

        # X_train shape: (train_len - time_steps) x (time_steps + train_set_len - 1)
        print('X_train last: ', X_train.shape)
        print('My expectation: 1641 x 75 x 1')
        print('-'*20)

        # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        return X_train, y_train

            
    def build(self, units, dropout, target_len):
        self.model = sq()

        # input
        self.model.add(LSTM(units=units, return_sequences = True, input_shape = (self.X_train.shape[1], 1)))
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


    def run(self, epoch, batch_size):
        # compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # fit model

        # no problem until here in case of shape of X and y train!
        print(self.X_train.shape)
        print(self.y_train.shape)
        self.model.fit(self.X_train, self.y_train, epochs = epoch, batch_size = batch_size)

    def predict(self, target_col, time_steps, scaler, product):

        import matplotlib.pyplot as plt

        dataset_X_total = pd.concat((self.train_set[:], self.test_set[:]), axis = 0, ignore_index=True)
        print('dataset_X_total: ',dataset_X_total.shape)
        dataset_X_total_without_target = dataset_X_total.drop([target_col], axis = 1)
        
        inputs = dataset_X_total.iloc[len(dataset_X_total) - len(self.test_set) - time_steps:, :].reset_index(drop=True) # last (time_steps) data in train set + test set
        print('inputs: ',inputs.shape)

        inputs_without_target = dataset_X_total_without_target.iloc[len(dataset_X_total) - len(self.test_set) - time_steps:]

        X_test = []
        for i in range(time_steps, time_steps + len(self.test_set)):
            time_steps_arr = inputs.loc[i-time_steps:i-1, [target_col]].values
            if i == time_steps: print('time_steps_arr: ',time_steps_arr.shape)
            features_arr = inputs_without_target.iloc[i-1:i, :].values.reshape(-1, 1)
            if i == time_steps: print('features_arr: ',features_arr.shape)
            X_test.append(np.concatenate((time_steps_arr, features_arr), axis=0))
            
        X_test = np.array(X_test)
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], 1))

        predicted_prices = self.model.predict(X_test)

        predicted_prices_inv = scaler.inverse_transform(predicted_prices)
        test_set_target_inv = scaler.inverse_transform(self.test_set[target_col].values.reshape(-1,1))

        plt.figure(figsize=(8,4))
        plt.plot(test_set_target_inv, color='deepskyblue', label = f'{product} Real Price')
        plt.plot(predicted_prices_inv, color='tomato', label = f'{product} Estimated Price')
        plt.title(f'{product} Market Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        print('\n', 'Process is succeeded! Model results are in shown!')