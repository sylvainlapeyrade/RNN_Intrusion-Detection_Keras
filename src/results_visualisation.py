import numpy as np
import pandas as pd
from kdd_processing import kdd_encoding
from unsw_processing import unsw_encoding
from tensorflow._api.v1.keras.models import load_model
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_score,
                             roc_curve, recall_score, auc)

# Allows tensorflow to run multiple sessions (Multiple learning simultaneously)
# Comment the 4 following lines if causing issues
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

data_path = './data/'
model_path = './models/'

# ***** REFERENCES PARAMETERS *****
params = {'train_data': 494021, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standarscaler'}

# ***** VARIABLE PARAMETERS *****
params_var = {'encoder': ['standardscaler', 'labelencoder',
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'batch_size': [128, 256, 512, 1024, 2048],
              # 'features_nb': [4, 8, 41],
              # 'train_data': [494021, 4898431, 125973, 25191, 82333],
              # 'cell_type': ['CuDNNLSTM', 'CuDNNGRU', 'SimpleRNN'],
              }

dataset = 'kdd'
model_name = '494021_4_mse_nadam_sigmoid_1_128_1024_0.2' + \
    '_CuDNNLSTM_standardscaler_1563377470.7797422'

if dataset == 'kdd':
    x_train, x_test, y_train, y_test = kdd_encoding(params)
elif dataset == 'unsw':
    x_train, x_test, y_train, y_test = unsw_encoding(params)

model = load_model(model_path+model_name)
model.summary()

score = model.evaluate(x_test, y_test, batch_size=params['batch_size'])
y_pred = model.predict(x_test, batch_size=params['batch_size'])

print('\nMatrice de confusion:')
confusion_matrix = confusion_matrix(
    y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(confusion_matrix)

FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + FP)

print('\nTPR:')
print(TP/(TP+FN))
print('\nFPR:')
print(FP/(FP+TN))

# Cost Matrix as presented in Staudemeyer article
cost_matrix = [[0, 1, 2, 2, 2],
               [1, 0, 2, 2, 2],
               [2, 1, 0, 2, 2],
               [4, 2, 2, 0, 2],
               [4, 2, 2, 2, 0]]

tmp_matrix = np.zeros((5, 5))

for i in range(5):
    for j in range(5):
        tmp_matrix[i][j] = confusion_matrix[i][j] * cost_matrix[i][j]

# The average cost is (total cost / total number of classifications)
print('\nCost:')
print(tmp_matrix.sum()/confusion_matrix.sum())

print('\nAUC:')
print(roc_auc_score(y_true=y_test, y_score=y_pred, average=None))

print('\nPrecision:')
print(precision_score(y_true=y_test.argmax(axis=1),
                      y_pred=y_pred.argmax(axis=1), average=None))
