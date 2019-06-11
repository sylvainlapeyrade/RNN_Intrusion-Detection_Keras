from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
from tensorflow._api.v1.keras.models import load_model
import tensorflow as tf
import configparser as cp
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_score,
                             roc_curve, recall_score, auc)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

config_parser = cp.ConfigParser()
config_parser.read('./config.ini')
x_train = np.load(config_parser.get('DataPath', 'x_train_path'))
y_train = np.load(config_parser.get('DataPath', 'y_train_path'))
x_test = np.load(config_parser.get('DataPath', 'x_test_path'))
y_test = np.load(config_parser.get('DataPath', 'y_test_path'))

model = load_model(
    './models/sigmoid_adam_50e_10p_4f_2l_128u_1559832818.8302574')
model.summary()

score = model.evaluate(x_test, y_test, batch_size=1024)
y_pred = model.predict(x_test, batch_size=1024)

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

# Matrice de coût tel que présenté dans l'article
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
