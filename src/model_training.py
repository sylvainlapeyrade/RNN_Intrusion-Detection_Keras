import configparser as cp
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

print("Chargement des donnees...")
# Chargements des donnees du fichiers de configuration
config_parser = cp.ConfigParser()
config_parser.read('./config.ini')

x_train = np.load(config_parser.get('DataPath', 'x_train_path'))
y_train = np.load(config_parser.get('DataPath', 'y_train_path'))
x_test = np.load(config_parser.get('DataPath', 'x_test_path'))
y_test = np.load(config_parser.get('DataPath', 'y_test_path'))
optimizer = config_parser.get('Parameters', 'optimizer')
epochs = config_parser.get('Parameters', 'epochs')
loss = config_parser.get('Parameters', 'loss')

path = '_' + optimizer + '_' + epochs + 'e_10p_8f_2l_128u_' + str(time())

# Réglages Tensorboard
log_path = config_parser.get('DataPath', 'log_path')
print("Fichier de log Tensorboard: ", log_path+path)
tensorboard = TensorBoard(log_path+path)

print("Creation du modele...")
model = Sequential()  # Pile lineaire de couches

# Add : couche qui ajoute des inputs
# Dropout : randomly set a fraction rate of input units to 0 at
#  each update during training time, which helps prevent overfitting.
model.add(CuDNNLSTM(units=128, input_shape=(x_train.shape[1:]),
                    return_sequences=True))
model.add(Dropout(rate=0.2))

model.add(CuDNNLSTM(units=128))
model.add(Dropout(rate=0.1))

model.add(Dense(units=5, activation='sigmoid'))

# Loss the objective that the model will try to minimize
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

print("Entrainement du modele...")

model.fit(x_train, y_train, epochs=int(epochs), validation_data=(x_test, y_test),
          use_multiprocessing=True, callbacks=[tensorboard])

y_pred = model.predict(x_test, verbose=1)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print(matrix)

# Réglages sauvegarde modèle
model_path = config_parser.get('DataPath', 'model_path')
model.save(model_path+path)
print("Modele enregistre: ", model_path+path)
