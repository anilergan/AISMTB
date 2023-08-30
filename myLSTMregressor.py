class my_RNN_LSTM_Regressor:

    def __init__(self, x, y, test_num, pred_num, units=50, dropout=0.2, epoch=50, batch_size=32, predict=False, scaler=None):

        df = pd.concat([x, y], axis = 1)

        self.train_set, self.test_set = self.train_test_split(df, test_num)

        self.X_train, self.y_train = self.pred_split(pred_num, x.shape[1])

        self.build(units, dropout, x.shape[1])

        self.run(epoch, batch_size, x.shape[1])

        self.pred_arg_1 = y.column[0]
        self.pred_arg_2 = pred_num
        self.pred_arg_3 = scaler

        if predict:
            self.predict(self.pred_arg_1, self.pred_arg_2, self.pred_arg_3)
        
    def train_test_split(self, df, test_num):
        train_num = len(df) - test_num
        train_set = df.iloc[:train_num, :]
        test_set = df.iloc[train_num:, :]

        return train_set, test_set

    def pred_split(self, pred_num,x_col_len):
        X_train = []
        y_train = []
        for i in range(pred_num, self.train_set.shape[0]):
            X_train.append(self.train_set.iloc[i-pred_num:i,:x_col_len].values)
            y_train.append(self.train_set.iloc[i, x_col_len:].values)
        
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

    def predict(self,target_col, pred_num, scaler):

        import matplotlib.pyplot as plt

        df_total_target = pd.concat(self.train_set[target_col], self.test_set[target_col], axis = 0)
        inputs = df_total_target[len(df_total_target) - len(self.test_set) - pred_num:].values # last (pred_num) day data of train set + test set
        inputs = inputs.reshape(-1,1)

        X_test = []
        for i in range(pred_num, pred_num + len(self.test_set)):
            X_test.append(inputs[i-self.pred_num: self.pred_num, :])
        X_test = np.array(X_test)
        # inputs = train_set'in last (pred_num) values + test_set
        # so above code seperate x_test from inputs

        predicted_prices = self.model.predict(X_test)
        predicted_prices_inv = scaler.inverse_transform(predicted_prices)

        plt.plot(self.test_set[target_col], color='deepskyblue', label = 'Real Google Stack Price')
        plt.plot(predicted_prices_inv, color='lightskyblue', label = 'Predicted Google Stack Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()