from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class my_RNN_ANN_Mixed_Regressor():

    def __init__(self, x, y, x_rnn_cols,  test_size=0.2, time_steps=60, lstm_act='tanh', ann_act='relu', model_act='relu', model_units=32,  units=50, dropout=0.25, epoch=50, batch_size=32, predict=True, scaler, split_graph = 0, figsize=[8,4], product=''):
        
        self.x_R = x.loc[:, x_rnn_cols]
        self.x_A = x.drop(x_rnn_cols, axis=1)
        self.y = y

        target = y.columns[0]
        
        target_check = False
        for i in x_rnn_cols:
            if i == target: target_check = True
        
        if target_check: df = x
        else: df = pd.concat([x,y], axis=1)
         

        print('Process 1 : Splitting data as train and test')
        x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test  = self.train_test_split(df, test_size)
        
        print('Process 2: Rehaping RNN data by timesteps')
        X_R_train_shaped, X_R_test_shaped = self.rehape_rnn_data_by_timesteps(x_R_train, x_R_test, time_steps)

        print('Process 3.1: Building LSTM model')
        self.build_lstm(units, dropout, lstm_act)

        print('Process 3.2: Building ANN model')
        self.build_ann(units, dropout, ann_act)

        print('Process 3.3: Building the final model')
        self.build_model(model_units, model_act)

        print('Process 4: Running the Model')
        self.run(X_R_train_shaped, x_A_train, y_train, epoch, batch_size)

        if predict:
            print('Process 5.1: Estimating by the model')
            test_inv = scaler.inverse_transform(self.test_set[self.target_col].values.reshape(-1,1))

            pred_inv = self.predict(scaler, self.target_col, self.time_steps)

            print('Process 5.2: Visualizing of comparison reality and estimation')
            self.visualize(test_inv, pred_inv, figsize, product)
        

    def train_test_split(self, test_size):
        x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test = train_test_split(self.x_A, self.x_R, self.y, test_size=test_size)
        return x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test 

    def rehape_rnn_data_by_timesteps(self, x_R_train, x_R_test, time_steps):

        x_R_train_TS_list = []
        for i in range(time_steps, x_R_train.shape[0]):
            x_R_time_steps_array = x_R_train.iloc[i-time_steps:i-1, :].values
            x_R_train_TS_list.append(x_R_time_steps_array)

        
        inputs = self.x_R.iloc[len(self.x_R) - len(x_R_test) - time_steps].reset_index(drop=True)

        x_R_test_TS_list = []
        for i in range(time_steps, time_steps + len(x_R_test)):
            x_R_time_steps_array = x_R_test.iloc[i-time_steps:i-1, :].values
            x_R_test_TS_list.append(x_R_time_steps_array)
        
        # list to array
        x_R_train_TS_array, x_R_test_TS_array = np.array(x_R_train_TS_list), np.array(x_R_test_TS_list)

        # reshape arrays to put into lstm model
        x_R_train_lstm_array_reshaped, x_R_test_lstm_array_reshaped = np.reshape(x_R_train_TS_array, (x_R_train_TS_array.shape[0], x_R_train_TS_array.shape[1], 1)), np.reshape(x_R_test_TS_array, (x_R_test_TS_array.shape[0], x_R_test_TS_array.shape[1], 1))

        return x_R_train_lstm_array_reshaped, x_R_test_lstm_array_reshaped
            
    def build_lstm(self, units, dropout, lstm_act):
        self.model_lstm = Sequential()

        # input
        self.model_lstm.add(LSTM(units=units, activation= lstm_act, return_sequences = True, input_shape = (self.time_steps, 1)))
        self.model_lstm.add(Dropout(dropout))
        
        # first
        self.model_lstm.add(LSTM(units = units, activation= lstm_act, return_sequences = True))
        self.model_lstm.add(Dropout(dropout))

        # second
        self.model_lstm.add(LSTM(units = units, activation= lstm_act,return_sequences = True))
        self.model_lstm.add(Dropout(dropout))
        
        # output 
        self.model_lstm.add(Dense(units=1)) #target col len is always 1 for this class

    def build_ann(self, units, dropout, ann_act):
        self.model_ann = Sequential()

        # input
        self.model_ann.add(Dense(units=units, activation=ann_act,  input_shape = (self.x_A.shape[1], 1)))
        self.model_ann.add(Dropout(dropout))
        
        # first
        self.model_ann.add(Dense(units = units, activation=ann_act))
        self.model_ann.add(Dropout(dropout))

        # second
        self.model_ann.add(Dense(units = units, activation=ann_act))
        self.model_ann.add(Dropout(dropout))

        # output 
        self.model_ann.add(Dense(units=1)) #target col len is always 1 for this class

    def build_model(self, model_units, model_act):

        combined_input = concatenate([self.model_lstm.output, self.model_ann.output])
        
        output_ = Dense(units=model_units, activation=model_act)(combined_input)

        self.model = Model(inputs = [self.model_lstm.inputs, self.model_ann.inputs], output = output_)
        

    def run(self, x_R_train, x_A_train, y_train, epoch, batch_size):
        # compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # fit model
        self.model.fit([x_R_train, x_A_train], y_train, epoch=epoch, batch_size=batch_size )


    def predict(self, scaler, ):

        X_test = np.array(X_test)
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], 1))

        predicted_prices = self.model.predict(X_test)

        predicted_prices_inv = scaler.inverse_transform(predicted_prices)

        return predicted_prices_inv

    def visualize(self, test, pred, fig, product):

        plt.figure(figsize=(fig[0],fig[1]))
        plt.plot(pred, color='darkorange', label = f'{product} Real Price')
        plt.plot(test, color='darkorchid', label = f'{product} Estimated Price')
        plt.title(f'{product} Market Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        print('\n', 'Process is succeeded! Model results are in shown!')