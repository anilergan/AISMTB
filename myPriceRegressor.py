from keras.models import Sequential as sq
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class My_Price_Regressor():

    def __init__(self, df, target_col, test_num, time_steps, scaler, units=50, dropout=0.2, epoch=50, batch_size=32, predict=True, figsize=[18,6], split_graph = 0, product='', fit_verbose = 0):
        
        self.fit_verbose = fit_verbose
        self.target_col = target_col
        self.time_steps = time_steps 
        self.scaler_model = scaler
        
        self.x_sc, self.y_sc = self.preprocess(df, target_col, scaler)

        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(test_num)
        
        self.X_train, self.Y_train = self.train_set_ts_split()

        self.X_test = self.test_set_ts_split()

        self.build(units, dropout)

        self.run(epoch, batch_size)

        if predict:
            self.y_pred = self.test_predict()

            self.y_test_inv = self.scaler_model.inverse_transform(self.y_test)
            self.y_pred_inv = self.scaler_model.inverse_transform(self.y_pred)

            self.test_visualize(figsize, split_graph, product)

            self.test_error()
        

    def preprocess(self, df, target_col, scaler):
        self.x = df.loc[:,[target_col]]
        self.y = df.loc[:,[target_col]]
        # order is old to new
        
        # Scale
        x_sc = scaler.fit_transform(self.x)
        y_sc = scaler.fit_transform(self.y)

        return x_sc, y_sc


    def train_test_split(self, test_num):
        train_num = self.x_sc.shape[0] - test_num
        x_train = self.x_sc[:train_num]
        x_test = self.x_sc[train_num:]

        y_train = self.y_sc[:train_num]
        y_test = self.y_sc[train_num:]
        return x_train, x_test, y_train, y_test


    def train_set_ts_split(self):
        X_train = []
        Y_train = []
        for i in range(self.time_steps, self.x_train.shape[0]):

            x_train_ts = self.x_train[i-self.time_steps:i]
            y_train_ts = self.y_train[i]

            X_train.append(x_train_ts)
            Y_train.append(y_train_ts)

        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        return X_train, Y_train
    
    def test_set_ts_split(self):

        inputs = self.x_sc[len(self.x_sc) - len(self.x_test) - self.time_steps:] # last (time_steps) data in train set + test set

        X_test = []
        for i in range(self.time_steps, self.time_steps + len(self.x_test)):

            x_test_ts = inputs[i-self.time_steps:i]

            X_test.append(x_test_ts)
            
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        return X_test
            
    def build(self, units, dropout):
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
        self.model.add(Dense(units=1))


    def run(self, epoch, batch_size):
        # compile model
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        
        # fit model
        self.model.fit(self.X_train, self.Y_train, epochs = epoch, batch_size = batch_size, verbose=self.fit_verbose)


    def test_predict(self):
        y_pred = self.model.predict(self.X_test, verbose=self.fit_verbose)
        return y_pred



    def test_visualize(self, fig, split_graph, product):

        if split_graph == 0:
            plt.figure(figsize=(fig[0],fig[1]))
            plt.plot(self.y_test_inv, color='mediumblue', label = f'{product} Real Price')
            plt.plot(self.y_pred_inv, color='darkorchid', label = f'{product} Estimated Price')
            plt.title(f'{product} Market Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()
            plt.show()
        
        else:
            for i in range(split_graph):
                plt.figure(figsize=(fig[0],fig[1]))
                sel_bot = int((self.y_pred_inv.shape[0])-(self.y_pred_inv.shape[0]/split_graph)*(i+1))
                sel_top = int((self.y_pred_inv.shape[0])-(self.y_pred_inv.shape[0]/split_graph)*i)
                plt.plot(self.y_test_inv[sel_bot:sel_top], color='mediumblue', label = f'{product} Real Price')
                plt.plot(self.y_pred_inv[sel_bot:sel_top], color='darkorchid', label = f'{product} Estimated Price')
                plt.title(f'{product} Market Price Prediction Graph {i+1}/{split_graph}')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.show()  

    def test_error(self):
        real_price_change_perc = []
        real_price_change = []
        real_price = []
        for i in range(self.y_test_inv.shape[0] - 1):
            price = self.y_test_inv[i]
            price_change = self.y_test_inv[i+1] - self.y_test_inv[i]
            price_change_perc = price_change * 100 / self.y_test_inv[i]

            real_price.append(price)
            real_price_change.append(price_change)
            real_price_change_perc.append(price_change_perc)

        pred_price_change_perc = []
        pred_price_change = []
        pred_price = []
        for i in range(self.y_pred.shape[0] - 1):
            price = self.y_pred_inv[i]
            price_change = self.y_pred_inv[i+1] - self.y_pred_inv[i]
            price_change_perc = price_change * 100 / self.y_pred_inv[i]

            pred_price.append(price)
            pred_price_change.append(price_change)
            pred_price_change_perc.append(price_change_perc)

        error_price_change_perc = []
        error_price_change = []
        error_price = []
        for i in range(len(pred_price_change_perc)):

            error_price_change_perc.append(abs(real_price_change_perc[i] - pred_price_change_perc[i]))
            error_price_change.append(abs(real_price_change[i] - pred_price_change[i]))
            error_price.append(abs(real_price[i] - pred_price[i]))
        
        real_price, pred_price, error_price, real_price_change, pred_price_change, error_price_change, real_pc_perc, pred_pc_perc, error_pc_perc = np.array(real_price), np.array(pred_price), np.array(error_price), np.array(real_price_change), np.array(pred_price_change), np.array(error_price_change), np.array(real_price_change_perc), np.array(pred_price_change_perc), np.array(error_price_change_perc)
        
        # array_combined = np.concatenate([real_price, pred_price, error_price, real_price_change, pred_price_change, error_price_change, real_pc_perc, pred_pc_perc, error_pc_perc], axis = 1)

        # df_mpe = pd.DataFrame(data = array_combined, columns = ['real_price', 'pred_price', 'error_price', 'real_price_change', 'pred_price_change', 'error_price_change', 'real_pc_perc', 'pred_pc_perc', 'error_pc_perc'])

        # print('-'*20, '\n', df_mpe)
        # print('Mean Price Error: ', error_price.mean())
        # print('Mean Price Change Error: ', error_price_change.mean())
        # print('Mean Price Change Percentage Error: ', error_pc_perc.mean())

        self.error_array = np.array([error_price.mean(), error_price_change.mean(), error_pc_perc.mean()])


    def model_tune(self, test_num, epoch, time_steps, units, dropout, batch_size):

        total_process = len(test_num) * len(epoch) * len(time_steps) * len(units) * len(dropout) * len(batch_size)
        process_step = 1
        model_tune_array = np.array([])
        for tn in test_num:
            for e in epoch:
                for ts in time_steps:
                    for u in units:
                        for d in dropout:
                            for bs in batch_size:
                                print(f'Process Step: {process_step}/{total_process}')
                                self.time_steps = ts

                                self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(tn)
                                
                                self.X_train, self.Y_train = self.train_set_ts_split()

                                self.X_test = self.test_set_ts_split()

                                self.build(u, d)

                                self.run(e, bs)

                                self.y_pred = self.test_predict()

                                self.y_test_inv = self.scaler_model.inverse_transform(self.y_test)
                                self.y_pred_inv = self.scaler_model.inverse_transform(self.y_pred)

                                variables = np.array([tn, e, ts, u, d, bs])
                                
                                model_tune_array = np.append(model_tune_array, np.concatenate([self.error_array, variables]))

                                process_step += 1

        
        model_tune_array = model_tune_array.reshape(total_process,9)

        def model_tune_exhibit(table):
            table_df = pd.DataFrame(data=table, columns=['MPE', 'MPCE', 'MPCPE', 'test_num', 'epoch', 'time_steps', 'units', 'dropout', 'batch_size'])

            table_df['index'] = table_df.index
            table_df[table_df.columns[-1:].to_list() + table_df.columns[:-1].to_list()]

            table_df = table_df.sort_values(by= 'MPE', ascending=True)
            table_df['Score MPE'] = range(len(table_df))

            table_df = table_df.sort_values(by= 'MPCE', ascending=True)
            table_df['Score MPCE'] = range(len(table_df))

            table_df = table_df.sort_values(by= 'MPCPE', ascending=True)
            table_df['Score MPCPE'] = range(len(table_df))
            
            table_df['Total Score'] = table_df['Score MPE'] + table_df['Score MPCE'] + table_df['Score MPCPE'] 
            
            pd.set_option('display.width', 500)

            print('\n', '-|'*20, ' Total Score Top 3 ','-|'*20)
            print(table_df.sort_values(by='Total Score', ascending=True).iloc[1:4, :])
            print('\n', '-|'*20, ' Score MPE Top 3 ', '-|'*20)
            print(table_df.sort_values(by='Score MPE', ascending=True).iloc[1:4, :])
            print('\n', '-|'*20, ' Score MPCE Top 3 ', '-|'*20)
            print(table_df.sort_values(by='Score MPCE', ascending=True).iloc[1:4, :])
            print('\n', '-|'*20, ' Score MPCPE Top 3' , '-|'*20)
            print(table_df.sort_values(by='Score MPCPE', ascending=True).iloc[1:4, :])
                  
        model_tune_exhibit(model_tune_array)


    # def new_data_prediction(self, x_new, y_new):

    #     x, y = self.preprocess(x_new, y_new) 

    #     x = self.time_steps_split(self.x, self.time_steps)

    #     x
