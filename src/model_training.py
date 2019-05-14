import configparser as cp
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras.optimizers import Adam, SGD

#Chargements des donnees du fichiers de configuration
config_parser = cp.ConfigParser()
config_parser.read('./config.ini')

x_train = np.load(config_parser.get('DataPath', 'x_train_path'))
y_train = np.load(config_parser.get('DataPath', 'y_train_path'))
x_test = np.load(config_parser.get('DataPath', 'x_test_path'))
y_test = np.load(config_parser.get('DataPath', 'y_test_path'))
model_path = np.load(config_parser.get('DataPath', 'model_path'))

model = Sequential()
model.add(CuDNNLSTM(units=128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.1))

model.add(Dense(units=5, activation='sigmoid'))
model.add(Dropout(0.2))

model.add(Dense(units=5, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

y_pred = model.predict(x_test, verbose=1)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

model.save(model_path)