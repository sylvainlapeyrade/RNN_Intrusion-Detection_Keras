import configparser as cp
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
from keras.optimizers import Adam, SGD

print("Chargement des donnees...")
# Chargements des donnees du fichiers de configuration
config_parser = cp.ConfigParser()
config_parser.read('./config.ini')

x_train = np.load(config_parser.get('DataPath', 'x_train_path'))
y_train = np.load(config_parser.get('DataPath', 'y_train_path'))
x_test = np.load(config_parser.get('DataPath', 'x_test_path'))
y_test = np.load(config_parser.get('DataPath', 'y_test_path'))
log_path = config_parser.get('DataPath', 'log_path')

print("Creation du modele...")
model = Sequential()
model.add(CuDNNLSTM(units=128, input_shape=(x_train.shape[1:]),
                    return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(CuDNNLSTM(units=128))
model.add(Dropout(rate=0.1))

model.add(Dense(units=5, activation='sigmoid'))
model.add(Dropout(rate=0.2))

model.add(Dense(units=5, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam',
              metrics=['accuracy'])
model.summary()

print("Entrainement du modele...")

tensorboard = TensorBoard(log_dir=log_path+"/{}".format(time()))
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
          use_multiprocessing=True, callbacks=[tensorboard])

y_pred = model.predict(x_test, verbose=1)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

model_path = config_parser.get('DataPath', 'model_path')
model.save(model_path+"/{}".format(time()))
print("Modele enregistre a l'emplacement:", model_path)
