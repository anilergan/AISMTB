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
        self.X_train, self.y_train = self.time_steps_split(time_steps, x.shape[1], y.columns[0])

        print('Model is building...')
        self.build(units, dropout, x.shape[1])

        print('Model is running...')
        self.run(epoch, batch_size, x.shape[1])



        if predict:
            print('Estimating by model...')
            self.predict(self.pred_arg_1, self.pred_arg_2, self.pred_arg_3, self.pred_arg_4)
        
    def train_test_split(self, df, test_num):
        train_set = df.iloc[test_num:, :].reset_index(drop=True)
        test_set = df.iloc[:test_num, :].reset_index(drop=True)

        return train_set, test_set

    def time_steps_split(self, time_steps, x_col_len, target_col):
        X_train = []
        y_train = []
        for i in range(time_steps, self.train_set.shape[0]):
            X_train.append(self.train_set.iloc[i-time_steps:i,:].values)
            y_train.append(self.train_set.loc[i, [target_col]].values)

        X_train, y_train = np.array(X_train), np.array(y_train)
        return X_train, y_train

            
    def build(self, units, dropout, x_col_len):
        self.model = sq()

        # input
        self.model.add(LSTM(units=units, return_sequences = True, input_shape = (self.X_train.shape[1], x_col_len)))
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
        self.model.add(Dense(units=self.y_train.shape[1]))


    def run(self, epoch, batch_size, x_col_len):
        # compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # fit model
    
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], x_col_len))

        self.model.fit(self.X_train, self.y_train, epochs = epoch, batch_size = batch_size)

    def predict(self, target_col, time_steps, scaler, product):

        import matplotlib.pyplot as plt

        dataset_X_total = pd.concat((self.test_set[:], self.train_set[:], ), axis = 0)


        inputs = dataset_X_total.iloc[: len(self.test_set) + time_steps, :].values # {test set} + {last (time_steps) data in train set}

        X_test = []
        for i in range(len(self.test_set), 0, -1):
            X_test.append(inputs[i:i+time_steps, :])


  
        X_test_matrix = np.vstack(X_test) #.reshape(-1,1)

        # inputs = train_set's last (time_steps) values + test_set
        # so above code seperate x_test from inputs


        x_col_len = len(self.train_set.columns)

        # (batch_size, time_steps, features)
        X_test_matrix = np.reshape(X_test_matrix, (len(self.test_set), time_steps, x_col_len))
        

        predicted_prices = self.model.predict(X_test_matrix)
        predicted_prices_inv = scaler.inverse_transform(predicted_prices)
        
        test_set_target_inv = scaler.inverse_transform(self.test_set[target_col].values.reshape(-1,1))
    
        plt.plot(test_set_target_inv, color='blue', label = f'{product} Real Price')
        plt.plot(predicted_prices_inv, color='gold', label = f'{product} Estimated Price')
        plt.title(f'{product} Market Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        print('\n', 'Process is succeeded! Model results are in shown!')