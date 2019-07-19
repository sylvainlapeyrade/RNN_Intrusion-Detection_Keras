import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from kdd_processing import kdd_encoding
from unsw_processing import unsw_encoding

dataset = 'kdd'

params = {'epochs': 300, 'train_data': 494021, 'features_nb': 4,
          'loss_fct': 'mse', 'optimizer': 'nadam',
          'activation_fct': 'sigmoid', 'layer_nb': 2, 'unit_nb': 128,
          'batch_size': 1024, 'dropout': 0.2, 'cell_type': 'CuDNNLSTM',
          'encoder': 'standarscaler'}

params_var = {'encoder': ['standardscaler', 'labelencoder',
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'batch_size': [128, 256, 512, 1024, 2048],
              # 'features_nb': [4, 8, 41],
              # 'train_data': [494021, 4898431, 125973, 25191, 82333],
              # 'cell_type': ['CuDNNLSTM', 'CuDNNGRU', 'SimpleRNN'],
              }


def load_new_data():
    if dataset == 'kdd':
        return kdd_encoding(params)
    elif dataset == 'unsw':
        return unsw_encoding(params)


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

score = model.evaluate(x_test, y_test, batch_size=params['batch_size'])
y_pred = model.predict(x_test, batch_size=params['batch_size'])

print('\nMatrice de confusion:')
confusion_matrix = confusion_matrix(
    y_test.argmax(axis=1), y_pred.argmax(axis=1))
print(confusion_matrix)

# model = SVC()
# model.fit(x_train, y_train)
# print(model.score(x_test, y_test))

model = RandomForestClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
