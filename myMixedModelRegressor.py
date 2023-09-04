from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class my_RNN_ANN_Mixed_Regressor():

    def __init__(self, x, y, x_rnn_cols, test_num, time_steps, scaler, units=50, dropout=0.25, epoch=50, batch_size=32, predict=False, figsize=[8,4], product=''):
        
        # Prediction arguments
        self.target_col = y.columns[0]
        self.time_steps = time_steps 
        self.scaler_model = scaler

        self.x_R = x.loc[:, x_rnn_cols]
        self.x_A = x.drop(x_rnn_cols, axis=1)
        
        target = y.columns[0]
        
        target_check = False
        for i in x_rnn_cols:
            if i == target: target_check = True
        
        if target_check: df = x
        else: df = pd.concat([x,y], axis=1)
         

        print('Processing: Train-Test Split...')
        self.train_set, self.test_set = self.train_test_split(df, test_num)
        
        print('Processing: Seperate Train Data to X and Y according to Time Steps...')
        self.X_train, self.y_train = self.time_steps_split(time_steps)

        print('Model is building...')
        target_len = y.shape[1]
        self.build(units, dropout)

        print('Model is running...')
        self.run(epoch, batch_size)

        if predict:
            print('Estimating by model...')
            test_inv = self.scaler_model.inverse_transform(self.test_set[self.target_col].values.reshape(-1,1))

            pred_inv = self.predict(self.target_col, self.time_steps)

            self.visualize(test_inv, pred_inv, figsize, product)
        

    def train_test_split(self, x_A, x_R, y, test_size):
        x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test = train_test_split(x_A, x_R, y, test_size=ts)
        return x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test 

    def rehape_rnn_data_by_timesteps(x_R_train, x_R_test, time_steps):
        x_R_train_reshaped = []
        x_R_test_reshaped = []

        for i in range(time_steps, x_R_train.shape[0]):
            X_R_train.iloc[i+1: ]




            
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
        

    def run(self, epoch, batch_size):
        # compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # fit model
        model.fit([])


        # no problem until here in case of shape of X and y train!
        print(self.X_train.shape)
        print(self.y_train.shape)
        self.model.fit(self.X_train, self.y_train, epochs = epoch, batch_size = batch_size)

    def predict(self):

        dataset_X_total = pd.concat((self.train_set[:], self.test_set[:]), axis = 0, ignore_index=True)
        print('dataset_X_total: ',dataset_X_total.shape)
        dataset_X_total_without_target = dataset_X_total.drop([self.target_col], axis = 1)
        
        inputs = dataset_X_total.iloc[len(dataset_X_total) - len(self.test_set) - self.time_steps:, :].reset_index(drop=True) # last (time_steps) data in train set + test set
        print('inputs: ',inputs.shape)

        inputs_without_target = dataset_X_total_without_target.iloc[len(dataset_X_total) - len(self.test_set) - self.time_steps:]

        X_test = []
        for i in range(self.time_steps, self.time_steps + len(self.test_set)):
            time_steps_arr = inputs.loc[i-self.time_steps:i-1, [self.target_col]].values
            if i == self.time_steps: print('time_steps_arr: ',time_steps_arr.shape)
            features_arr = inputs_without_target.iloc[i-1:i, :].values.reshape(-1, 1)
            if i == self.time_steps: print('features_arr: ',features_arr.shape)
            X_test.append(np.concatenate((time_steps_arr, features_arr), axis=0))
            
        X_test = np.array(X_test)
        # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], 1))

        predicted_prices = self.model.predict(X_test)

        predicted_prices_inv = self.scaler_model.inverse_transform(predicted_prices)

        return predicted_prices_inv

    def visualize(self, test, pred, fig, product):

        plt.figure(figsize=(fig[0],fig[1]))
        plt.plot(pred, color='deepskyblue', label = f'{product} Real Price')
        plt.plot(test, color='tomato', label = f'{product} Estimated Price')
        plt.title(f'{product} Market Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        print('\n', 'Process is succeeded! Model results are in shown!')