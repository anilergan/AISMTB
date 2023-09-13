from keras.models import Sequential as sq
from tensorflow import float32, convert_to_tensor
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from time import time


class My_Combined_LSTM_Classifier():

    def __init__(self, x, y, test_num, time_steps=10,  hidden_layer_num = 3, units=50, dropout=0.2, epoch=10, batch_size=32, model_activation = 'relu', predict=True, fit_verbose = 1):
        self.X = x[:, :-1]
        self.X_change = x[:, -1:]
        self.Y = y

        self.fit_verbose = fit_verbose
        self.time_steps = time_steps 
        
        self.x_train, self.x_test, self.x_change_test, self.y_train, self.y_test = self.train_test_split(test_num)
        
        self.X_train, self.Y_train = self.train_set_ts_split()

        self.X_test = self.test_set_ts_split()

        self.build(hidden_layer_num, units, dropout, model_activation)

        self.run(epoch, batch_size)

        if predict:
            test_accuracy, train_accuracy, profit = self.evaluate()
            print('Train Accuracy: ', train_accuracy)
            print('Test Accuracy:', test_accuracy)
            print(f'Profit: {profit}%')

            # self.y_test_inv = self.scaler_model.inverse_transform(self.y_test)
            # self.y_pred_inv = self.scaler_model.inverse_transform(self.y_pred)

            # self.test_error()
        
    def train_test_split(self, test_num):
        train_num = self.X.shape[0] - test_num

        x_train = self.X[:train_num]
        x_test = self.X[train_num:]

        x_change_test = self.X_change[train_num:]

        y_train = self.Y[:train_num]
        y_test = self.Y[train_num:]
        return x_train, x_test, x_change_test, y_train, y_test


    def train_set_ts_split(self):
        X_train = []
        Y_train = []
        for i in range(self.time_steps, self.x_train.shape[0]):
            x_train_ts =  self.x_train[i-self.time_steps:i, :]
            y_train_ts = self.y_train[i, :]
            X_train.append(x_train_ts)
            Y_train.append(y_train_ts)

        X_train, Y_train = np.array(X_train), np.array(Y_train)
        # data-length x features x 

        return X_train, Y_train
    
    def test_set_ts_split(self):

        inputs = self.X[len(self.X) - len(self.x_test) - self.time_steps:] # last (time_steps) data in train set + test set

        X_test = []
        for i in range(self.time_steps, self.time_steps + len(self.x_test)):
            x_test_ts = inputs[i-self.time_steps:i, :]
            X_test.append(x_test_ts)
            
        X_test = np.array(X_test)
        return X_test
            
    def build(self, hidden_layer, units, dropout, model_activation):
        self.model = sq()

        # input
        self.model.add(LSTM(units=units, return_sequences = True, input_shape = (self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(dropout))
        
        # hidden layers
        for i in range(hidden_layer-1):
            self.model.add(LSTM(units = units, activation=model_activation, return_sequences = True))
            self.model.add(Dropout(dropout))


        # last hidden layer
        self.model.add(LSTM(units = units, activation=model_activation, return_sequences = False))
        self.model.add(Dropout(dropout))
        
        # output
        self.model.add(Dense(units=self.Y.shape[1], activation='sigmoid'))


    def run(self, epoch, batch_size):
        # compile model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        
        # fit model
        self.X_train = convert_to_tensor(self.X_train, dtype=float32)
        self.Y_train = convert_to_tensor(self.Y_train, dtype=float32)

        self.model_history = self.model.fit(self.X_train, self.Y_train, epochs = epoch, batch_size = batch_size, verbose=self.fit_verbose)


    def evaluate(self):
        self.X_test = convert_to_tensor(self.X_test, dtype=float32)
        self.y_pred = self.model.predict(self.X_test, verbose=self.fit_verbose)
        # # Düz tahminleri kategorik etiketlere dönüştür
        y_pred_category = np.argmax(self.y_pred, axis=1)
        y_test_category = np.argmax(self.y_test, axis=1)

        # Accuracy hesapla
        test_accuracy = accuracy_score(y_test_category, y_pred_category)
        test_accuracy = round(test_accuracy * 100, 2)

        train_accuracy = self.model_history.history['accuracy'][-1]
        train_accuracy = round(train_accuracy * 100, 2) 

        investment_profit = self.test_investment(y_pred_category)

        # con_matrix = confusion_matrix(y_test_category, y_pred_category)

        return test_accuracy, train_accuracy, investment_profit


    def test_investment(self, y_pred):
        pred_df = pd.DataFrame(data = 0, index=range(len(y_pred)), columns=['bear','volatile', 'bull', 'change', 'investment'])

        for i in range(len(y_pred)):
            if y_pred[i] == 0:
                pred_df.iloc[i, 0] = 1
            elif y_pred[i] == 1: 
                pred_df.iloc[i, 1] = 1
            elif y_pred[i] == 2:
                pred_df.iloc[i, 2] = 1
        for i in range(len(self.x_change_test)):
            pred_df.iloc[i, 3] = self.x_change_test[i]

        
        pred_df.iloc[:, 4] = 100
        
        for i in range(0, len(pred_df)-1):
            if pred_df.loc[i, 'bull'] == 1:
                pred_df.loc[i+1, 'investment'] = (pred_df.loc[i, 'investment'] * pred_df.loc[i+1, 'change'] / 100) + pred_df.loc[i-1, 'investment'] #inside
            elif pred_df.loc[i, 'volatile'] == 1: pred_df.loc[i+1, 'investment'] = pred_df.loc[i, 'investment'] #not insid
        
            elif pred_df.loc[i, 'bear'] == 1: pred_df.loc[i+1, 'investment'] = pred_df.loc[i, 'investment'] #not inside

        pred_df.loc[:,['bear', 'volatile', 'bull', 'change', 'investment']]

        investment_profit = round(pred_df.iloc[-1, -1] - pred_df.iloc[0, -1], 2)


        return investment_profit

        
        


    def model_tune(self, test_num, epoch, time_steps, units, dropout, hidden_layer_num, model_activation, batch_size):

        total_process = len(test_num) * len(epoch) * len(time_steps) * len(units) * len(dropout) * len(hidden_layer_num) * len(model_activation) * len(batch_size)
        process_step = 1
        model_tune_array = np.array([])

        print('THE MODEL IS TUNING!\nSEE U AN ETERNITY LATER *_*\n')
        print(f'Total Progress is creating by {total_process} steps.\n')
        for tn in test_num:
            for e in epoch:
                for ts in time_steps:
                    for u in units:
                        for d in dropout:
                            for hln in hidden_layer_num:
                                for ma in model_activation:
                                        for bs in batch_size:
                                            
                                            start_time = time()

                                            self.time_steps = ts

                                            self.x_train, self.x_test, self.x_change_test, self.y_train, self.y_test = self.train_test_split(tn)
                                            
                                            self.X_train, self.Y_train = self.train_set_ts_split()

                                            self.X_test = self.test_set_ts_split()

                                            self.build(hln, u, d, ma)

                                            self.run(e, bs)

                                            test_acc, train_acc, profit = self.evaluate()
                                            results = np.array([profit, test_acc, train_acc])

                                            variables = np.array([tn, e, ts, u, d, hln, ma, bs])
                                            
                                            model_tune_array = np.append(model_tune_array, np.concatenate([results, variables]))

                                            end_time = time()
                                
                                            #################################

                                            def convert_second_to_minute(sure):
                                                minute = int(sure / 60)
                                                second = int(sure % 60)
                                                return f"{minute} min {second} sec"
                                            
                                            if (end_time-start_time) >= 60 :
                                                time_str = convert_second_to_minute((end_time-start_time))
                                            
                                            else: time_str = f"{round((end_time-start_time), 1)} sec"
                                                

                                            print(f'Tune Step: {process_step}/{total_process} is done! It took {time_str}')

                                            process_step += 1

        
        model_tune_array = model_tune_array.reshape(total_process,model_tune_array.shape[1])

        def model_tune_exhibit(table):
            table_df = pd.DataFrame(data=table, columns=['profit(%)', 'test_accuracy(%)', 'train_accuracy(%)', 'test_num', 'epoch', 'time_steps', 'units', 'dropout', 'hidden_layer_num', 'model_activation', 'batch_size'])

            table_df['index'] = table_df.index
            table_df = table_df[table_df.columns[-1:].to_list() + table_df.columns[:-1].to_list()]
            
            pd.set_option('display.width', 500)

            print('\n', '-|'*17, ' Top 10 Profit and Variables ','-|'*17)
            print(table_df.sort_values(by= 'profit(%)', ascending=False).iloc[:10, :])
             
        model_tune_exhibit(model_tune_array)
