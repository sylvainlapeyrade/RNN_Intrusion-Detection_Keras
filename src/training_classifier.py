import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc)

from kdd_processing import kdd_encoding
from unsw_processing import unsw_encoding

# train_data: Number of rows in training dataset (see processing files)
# features_nb: Number of features kept as input (see processing files)
# batch_size: Number of elements observed before updating weights
# encoder: Encoding performed (see processing files)
# dataset: Processing file to be called ['kdd', 'unsw']
params = {'train_data': 494021, 'features_nb': 4, 'batch_size': 1024,
          'encoder': 'standarscaler', 'dataset': 'kdd'}

params_var = {'encoder': ['standardscaler', 'labelencoder',
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'batch_size': [128, 256, 512, 1024, 2048],
              # 'features_nb': [4, 8, 41],
              # 'train_data': [494021, 4898431, 125973, 25191],
              }


# Encode dataset and return : x_train, x_test, y_train, y_test
def load_data():
    if params['dataset'] == 'kdd':
        return kdd_encoding(params)
    elif params['dataset'] == 'unsw':
        return unsw_encoding(params)


def print_results_classifier(model):
    print(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    print('\nConfusion Matrix:')
    print(confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1)))


def MLPClassifier_train():
    print('*** MULTI LAYER PERCEPTRON ***')
    model = MLPClassifier(warm_start=True, verbose=True,
                          learning_rate='adaptive', early_stopping=True,
                          batch_size=params['batch_size'])
    model.fit(x_train, y_train)
    print_results_classifier(model)


def RandomForestClassifier_train():
    print('*** RANDOM FOREST ***')
    model = RandomForestClassifier(verbose=1, warm_start=True)
    model.fit(x_train, y_train)
    print_results_classifier(model)


def DecisionTreeClassifier_train():
    print('*** DECISION TREE ***')
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    print_results_classifier(model)


def KNeighborsClassifier_train():
    print('*** K-NEAREST NEIGHBORS ***')
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    print_results_classifier(model)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()

    MLPClassifier_train()
    RandomForestClassifier_train()
    DecisionTreeClassifier_train()
    KNeighborsClassifier_train()  # Can take a long time
