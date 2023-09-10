from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class My_Ichimoku_Price_Predictor():

    def __init__(self, x, y, scaler,  test_size=0.2, time_steps=60, lstm_act='tanh', ann_act='relu', model_act='relu', model_units=32,  units=50, dropout=0.25, epoch=50, batch_size=32, predict=True, split_graph = 0, figsize=[8,4], product=''):
        
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
        x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test  = self.data_tt_split(test_size, time_steps)
        
        print('Process 2: Rehaping RNN data by timesteps')
        x_R_train, x_R_test = self.rehape_rnn_data_by_timesteps(x_R_train, x_R_test, time_steps)

        print('Process 3.1: Building LSTM model')
        self.build_lstm(units, dropout, lstm_act, time_steps)

        print('Process 3.2: Building ANN model')
        self.build_ann(units, dropout, ann_act)

        print('Process 4.1: Running the Models')
        self.run_models(x_A_train, x_R_train, y_train, epoch, batch_size, time_steps)

        self.predict_models(x_A_test, x_R_test)

        print('Process 4.2: Creating/Running the Hybrid Model')
        self.hybrid(x_A_train, x_R_train, y_train, epoch)

        if predict:
            print('Process 5.1: Estimating by the hybrid model')
            test_inv = scaler.inverse_transform(y_test).reshape(-1,1)
            pred_inv = self.predict_hybrid(x_A_test, X_R_test, scaler)

            print('Process 5.2: Visualizing of comparison reality and estimation')
            self.visualize(test_inv, pred_inv, figsize, product)
        

    def data_tt_split(self, test_size, time_steps):
        x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test = train_test_split(self.x_A, self.x_R, self.y, test_size=test_size)
        print('x_A_train: ',x_A_train.shape)
        print('x_A_test: ',x_A_test.shape)
        print('y_train: ',y_train.shape)
        print('y_test: ',y_test.shape)
        return x_A_train, x_A_test, x_R_train, x_R_test, y_train, y_test 

    def rehape_rnn_data_by_timesteps(self, x_R_train, x_R_test, time_steps):

        x_R_train_TS_list = []
        for i in range(time_steps, x_R_train.shape[0]):
            x_R_time_steps_array = x_R_train.iloc[i-time_steps:i, :].values
            x_R_train_TS_list.append(x_R_time_steps_array)


        inputs = self.x_R.iloc[(len(self.x_R) - len(x_R_test) - time_steps):, :].reset_index(drop=True)
        x_R_test_TS_list = []
        for i in range(time_steps, time_steps + len(x_R_test)):
            x_R_time_steps_array = inputs.iloc[i-time_steps:i, :].values
            x_R_test_TS_list.append(x_R_time_steps_array)
        
        # list to array
        x_R_train_TS_array, x_R_test_TS_array = np.array(x_R_train_TS_list), np.array(x_R_test_TS_list)

        # reshape arrays to put into lstm model

        x_R_train_lstm_array_reshaped, x_R_test_lstm_array_reshaped = np.reshape(x_R_train_TS_array, (x_R_train_TS_array.shape[0], x_R_train_TS_array.shape[1], 1)), np.reshape(x_R_test_TS_array, (x_R_test_TS_array.shape[0], x_R_test_TS_array.shape[1], 1))

        print('x_R_train: ',x_R_train_lstm_array_reshaped.shape)
        print('x_R_test: ',x_R_test_lstm_array_reshaped.shape)

        return x_R_train_lstm_array_reshaped, x_R_test_lstm_array_reshaped
            
    def build(self, units, dropout, lstm_act, time_steps):
        self.model_lstm = Sequential()

        # input
        self.model_lstm.add(LSTM(units=units, activation= lstm_act, return_sequences = True, input_shape = (time_steps, 1)))
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

        

    def run_models(self, x_A_train, x_R_train, y_train, epoch, batch_size, time_steps):

        # compile both model
        self.model_lstm.compile(optimizer='adam', loss='mean_squared_error')
        self.model_ann.compile(optimizer='adam', loss='mean_squared_error')

        # train both model 
        self.model_lstm.fit(x_R_train.reshape(-1, x_R_train.shape[1], 1), y_train[time_steps:], epochs=epoch, batch_size=batch_size) 

        self.model_ann.fit(x_A_train, y_train, epochs=epoch, batch_size=batch_size) 


    def predict_models(self, x_A_test, x_R_test):
        pred_A = self.model_ann.predict(x_A_test)
        pred_R = self.model_lstm.predict(x_R_test.reshape(-1, x_R_test.shape[1], 1))
        print('pred_A shape: ', pred_A.shape)
        print('pred_R shape: ', pred_R.shape)
        return pred_A, pred_R


    def hybrid(self, pred_A, pred_R, x_A_test, x_R_test, y_test, epoch):
        ensemble_input = concatenate([pred_A, pred_R])
        ensemble_output = Dense(1)(ensemble_input)

        # create the hybrid model
        print('lstm output shape: ', self.model_lstm.layers[-1].output.shape)
        print('ann output shape: ', self.model_ann.layers[-1].output.shape)

        self.hybrid_model = Model(inputs=[self.model_ann.input, self.model_lstm.input], outputs=ensemble_output)

        # compile the hybrid model
        self.hybrid_model.compile(optimizer='adam', loss='mean_squared_error')

        # train the hybrid model
        self.hybrid_model.fit([x_R_test.reshape(-1, x_R_test.shape[1], 1), x_A_test], y_test, epochs=epoch, batch_size=32)

    def predict_hybrid(self, x_A_test, x_R_test, sc):

        predictions = self.hybrid_model.predict([x_R_test.reshape(-1, x_R_test.shape[1], 1), x_A_test])
        print('predictions shape:', predictions.shape)
        predictions = sc.inverse_transform(predictions)

        return predictions


    def visualize(self, test, pred, fig, product):

        plt.figure(figsize=(fig[0],fig[1]))
        plt.plot(pred, color='darkorange', label = f'{product} Real Price')
        plt.plot(test, color='darkorchid', label = f'{product} Estimated Price')
        plt.title(f'{product} Market Price Prediction')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        
        print('\n', 'Process is succeeded! Model results are in shown!')