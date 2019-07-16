import pandas as pd
import numpy as np
import os
from time import time
from kdd_data_processing import kdd_processing
from unsw_data_processing import unsw_processing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

data_path = './data/'
results_path = "./results/"
dataset = 'kdd'

params = {'epochs': 300, 'train_data': 25191, 'features_nb': 4,
          'loss_fct': 'mse', 'optimizer': 'nadam',
          'activation_fct': 'sigmoid', 'layer_nb': 2, 'unit_nb': 128,
          'batch_size': 1024, 'dropout': 0.2, 'cell_type': 'CuDNNLSTM',
          'encoder': 'standarscaler'}


def load_new_data():
    if dataset == 'kdd':
        kdd_processing(params)
    elif dataset == 'unsw':
        unsw_processing(params)

    x_train = np.load(data_path + dataset + '_x_train.npy')
    y_train = np.load(data_path + dataset + '_y_train.npy')
    x_test = np.load(data_path + dataset + '_x_test.npy')
    y_test = np.load(data_path + dataset + '_y_test.npy')

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = load_new_data()

model = MLPClassifier(solver='adam', alpha=0.00001,
                      activation='logistic',
                      hidden_layer_sizes=(5, 3),
                      random_state=1, verbose=True)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

model = KNeighborsClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

# model = SVC()
# model.fit(x_train, y_train)
# print(model.score(x_test, y_test))

model = RandomForestClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
