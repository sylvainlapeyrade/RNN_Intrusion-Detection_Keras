import configparser as cp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

config_parser = cp.ConfigParser()
config_parser.read('./config.ini')
x_train = np.load(config_parser.get('DataPath', 'x_train_path'))
y_train = np.load(config_parser.get('DataPath', 'y_train_path'))
x_test = np.load(config_parser.get('DataPath', 'x_test_path'))
y_test = np.load(config_parser.get('DataPath', 'y_test_path'))

model = load_model('model.h5')
model.summary()

score = model.evaluate(x_test, y_test)
# tn, fp, fn, tp = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)).ravel()

y_pred = model.predict(x_test, verbose=1)
matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)

# print("\nTest score:", score[0])
# print('Test accuracy:', score[1])

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()