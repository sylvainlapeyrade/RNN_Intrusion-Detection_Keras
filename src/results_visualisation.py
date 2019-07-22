import numpy as np
import tensorflow
from keras.models import load_model
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc)

from kdd_processing import kdd_encoding
from unsw_processing import unsw_encoding

# ***** REFERENCES PARAMETERS *****
params = {'train_data': 494021, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standarscaler',
          'dataset': 'kdd'}

model_name = './models/' + '494021_4_mse_nadam_sigmoid_1_128_1024' + \
    '_0.2_CuDNNLSTM_standarscaler_1562685990.8704927st'


# Encode dataset and return : x_train, x_test, y_train, y_test
def load_data():
    if params['dataset'] == 'kdd':
        x_train, x_test, y_train, y_test = kdd_encoding(params)
    elif params['dataset'] == 'unsw':
        x_train, x_test, y_train, y_test = unsw_encoding(params)

    # Reshape the inputs in the accepted model format
    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test


# Print information on the results
def print_results(params, model, x_train, x_test, y_train, y_test):
    print('Val loss and acc:')
    print(model.evaluate(x_test, y_test, params['batch_size']))

    y_pred = model.predict(x_test, params['batch_size'])

    print('\nConfusion Matrix:')
    conf_matrix = confusion_matrix(y_test.argmax(axis=1),
                                   y_pred.argmax(axis=1))
    print(conf_matrix)

    FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  # False Positive
    FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)  # False Negative
    TP = np.diag(conf_matrix)  # True Positive
    TN = conf_matrix.sum() - (FP + FN + FP)  # True Negative

    print('\nTPR:')  # True Positive Rate
    # Portion of positive instances correctly predicted positive
    print(TP / (TP + FN))

    print('\nFPR:')  # False Positive Rate
    # Portion of negative instances wrongly predicted positive
    print(FP / (FP + TN))

    # Cost Matrix as presented in Staudemeyer article
    cost_matrix = [[0, 1, 2, 2, 2],
                   [1, 0, 2, 2, 2],
                   [2, 1, 0, 2, 2],
                   [4, 2, 2, 0, 2],
                   [4, 2, 2, 2, 0]]

    tmp_matrix = np.zeros((5, 5))

    for i in range(5):
        for j in range(5):
            tmp_matrix[i][j] = conf_matrix[i][j] * cost_matrix[i][j]

    # The average cost is (total cost / total number of classifications)
    print('\nCost:')
    print(tmp_matrix.sum()/conf_matrix.sum())

    print('\nAUC:')  # Average Under Curve
    print(roc_auc_score(y_true=y_test, y_score=y_pred, average=None))

    print('\nPrecision:')  # Probability an instance gets correctly predicted

    print(precision_score(y_true=y_test.argmax(axis=1),
                          y_pred=y_pred.argmax(axis=1), average=None))


if __name__ == "__main__":
    # Allows tensorflow to run multiple sessions and learning simultaneously
    # Comment the 3 following lines if causing issues
    # config = tensorflow.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tensorflow.Session(config=config)

    model = load_model(model_name)
    model.summary()

    x_train, x_test, y_train, y_test = load_data()
    print_results(params, model, x_train, x_test, y_train, y_test)
