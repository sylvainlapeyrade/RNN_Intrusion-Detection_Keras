import configparser as cp
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM
from keras.optimizers import Adam, SGD


model = Sequential()  # Pile linèaire de couches

# CuDNNLSTM: Couche LSTM GPU, Units: Nombres de cellules,
#  input_shape: taille entrées, return_sequences: résultats après chaque noeud
model.add(CuDNNLSTM(units=128, input_shape=(, 8, 1), return_sequences=True))
model.add(Dropout(rate=0.2))  # Dropout: Jette une partie des entrées

model.add(CuDNNLSTM(units=128))  # Add: Ajoute une couche LSTM supplémentaire
model.add(Dropout(rate=0.1))

# Couche dense: chaque unité de la couche n-1 est connecté avec chaque de n+1
model.add(Dense(units=5, activation='sigmoid'))  # Nb sortie et fct activation

# Loss: objectif que le modèle va essayer de minimiser
# Optimizer: optimise le ratio d'entrainement en fonction du modèle
# Metrics: mesure utilisé pour évaluer notre modèle

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Entrainement du modele: x: entrées, y:sorties, epochs : Nb itération
# Validation_data: données de test
model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test))
