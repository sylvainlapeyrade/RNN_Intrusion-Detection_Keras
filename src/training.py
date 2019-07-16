from keras.backend.tensorflow_backend import set_session
from keras.layers import Dense, Dropout, CuDNNLSTM, CuDNNGRU, RNN, LSTM
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import numpy as np
import os
from time import time
from kdd_data_processing import kdd_processing
from unsw_data_processing import unsw_processing
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

csv_values = ['epochs', 'acc', 'loss', 'val_acc', 'val_loss', "train_data",
              "features_nb", 'loss_fct', 'optimizer', 'activation_fct',
              'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type',
              'encoder']
csv_best_res = ['param', 'value', 'min_mean_val_loss']

data_path = './data/'
results_path = "./results/"

# ***** REFERENCES PARAMETERS *****
params = {'epochs': 2, 'train_data': 494021, 'features_nb': 4,
          'loss_fct': 'mse', 'optimizer': 'rmsprop',
          'activation_fct': 'sigmoid', 'layer_nb': 2, 'unit_nb': 128,
          'batch_size': 1024, 'dropout': 0.2, 'cell_type': 'CuDNNLSTM',
          'encoder': 'standardscaler'}

# ***** VARIABLE PARAMETERS *****
params_var = {'encoder': ['standardscaler', 'labelencoder',
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax',
                            'adadelta'],
              'activation_fct': ['sigmoid', 'softmax', 'relu', 'tanh'],
              'layer_nb': [1, 2, 3, 4],
              'unit_nb': [4, 8, 32, 64, 128, 256],
              'dropout': [0.1, 0.2, 0.3, 0.4],
              'batch_size': [512, 1024, 2048],
              # 'features_nb': [4, 8, 41],
              # 'train_data': [494021, 4898431, 125973, 25191],
              # 'cell_type': ['CuDNNLSTM', 'CuDNNGRU', 'RNN'],
              # 'dataset : ['kdd', 'unsw']
              }


training_number = 1
resultstocsv = False
resultstologs = False
dataset = 'kdd'
results_path += 'lstm_test/'
model_path = './models/'
logs_path = './logs/'

# ***** RESULTS CSV *****
if resultstocsv is True:
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    full_results_path = results_path+"full_results.csv"
    best_result_path = results_path+"best_result.csv"
    results_df = pd.DataFrame(columns=csv_values)
    results_df.to_csv(full_results_path, index=False)
    best_res_df = pd.DataFrame(columns=csv_best_res)
    best_res_df.to_csv(best_result_path, index=False)
    min_val_loss = 0.03

if resultstologs is True:
    file_name = str(params['train_data']) + '_' + str(params['features_nb']) +\
        '_' + params['loss_fct'] + '_' + params['optimizer'] + '_' +\
        params['activation_fct'] + '_' + str(params['layer_nb']) + '_' +\
        str(params['unit_nb']) + '_' + str(params['batch_size']) + '_' +\
        str(params['dropout']) + '_' + params['cell_type'] + '_' +\
        params['encoder'] + '_' + str(time())


def load_new_data():
    if dataset == 'kdd':
        kdd_processing(params)
    elif dataset == 'unsw':
        unsw_processing(params)

    x_train = np.load(data_path + dataset + '_x_train.npy')
    x_train = x_train.reshape([-1, x_train.shape[1], 1])
    y_train = np.load(data_path + dataset + '_y_train.npy')
    x_test = np.load(data_path + dataset + '_x_test.npy')
    x_test = x_test.reshape([-1, x_test.shape[1], 1])
    y_test = np.load(data_path + dataset + '_y_test.npy')
    return x_train, x_test, y_train, y_test


def train_model():
    if params['cell_type'] == 'CuDNNLSTM':
        cell = CuDNNLSTM
    elif params['cell_type'] == 'CuDNNGRU':
        cell = CuDNNGRU
    elif params['cell_type'] == 'RNN':
        cell = RNN
    elif params['cell_type'] == 'LSTM':
        cell = LSTM

    model = Sequential()
    for _ in range(params['layer_nb']-1):
        model.add(cell(units=params['unit_nb'],
                       input_shape=(x_train.shape[1:]), return_sequences=True))
        model.add(Dropout(rate=params['dropout']))

    if(params['layer_nb'] == 1):
        model.add(cell(units=params['unit_nb'], input_shape=x_train.shape[1:]))
        model.add(Dropout(rate=params['dropout']))
    else:
        model.add(cell(units=params['unit_nb']))
        model.add(Dropout(rate=params['dropout']))

    model.add(Dense(units=y_train.shape[1],
                    activation=params['activation_fct']))

    model.compile(loss=params['loss_fct'], optimizer=params['optimizer'],
                  metrics=['accuracy'])

    if resultstologs is True:
        save_model = ModelCheckpoint(filepath=model_path+file_name,
                                     monitor='val_acc', save_best_only=True)
        tensorboard = TensorBoard(logs_path+file_name)
        callbacks = [save_model, tensorboard]
    else:
        callbacks = None

    model.summary()

    return model.fit(x_train, y_train, epochs=params['epochs'], shuffle=True,
                     batch_size=params['batch_size'], verbose=2,
                     validation_data=(x_test, y_test), callbacks=callbacks)


if resultstocsv is True:
    def fill_dataframe(df, history, epoch):
        df = df.append({'epochs': epoch,
                        'acc':  history.history['acc'][epoch],
                        'loss': history.history['loss'][epoch],
                        'val_acc': history.history['val_acc'][epoch],
                        'val_loss': history.history['val_loss'][epoch],
                        'train_data': params['train_data'],
                        'features_nb': params['features_nb'],
                        'loss_fct': params['loss_fct'],
                        'optimizer': params['optimizer'],
                        'activation_fct': params['activation_fct'],
                        'layer_nb': params['layer_nb'],
                        'unit_nb': params['unit_nb'],
                        'batch_size': params['batch_size'],
                        'dropout': params['dropout'],
                        'cell_type': params['cell_type'],
                        'encoder': params['encoder']},
                       ignore_index=True)
        return df

    def get_min_mean_value(value):
        df = pd.read_csv(results_path+value+".csv", index_col=False)
        names = df[value].unique().tolist()

        df_loss = pd.DataFrame(columns=names)
        for i in range(len(names)):
            df_value_loss = df.loc[df[value] == names[i]]
            df_value_loss = df_value_loss.nsmallest(5, 'val_loss')
            df_loss[names[i]] = np.array(df_value_loss['val_loss'])
        return df_loss.mean().idxmin(), df_loss.mean().min()

    x_train, x_test, y_train, y_test = load_new_data()

    for value in params_var.keys():
        results_df.to_csv(results_path + value + ".csv", index=False)
        save_var = params[value]
        for var in params_var[value]:
            value_df = pd.DataFrame(columns=csv_values)
            params[value] = var
            if value == 'encoder' or value == 'train_data':
                x_train, x_test, y_train, y_test = load_new_data()
            for _ in range(training_number):
                history = train_model()
                for epoch in range(params['epochs']):
                    value_df = fill_dataframe(value_df, history, epoch)
            value_df.to_csv(full_results_path, header=False, index=False,
                            mode='a')
            value_df.to_csv(results_path+value+".csv", header=False,
                            index=False, mode='a')
        param_min_value, min_mean_loss = get_min_mean_value(value)
        if min_mean_loss < min_val_loss:
            params[value] = param_min_value
            min_val_loss = min_mean_loss
        else:
            params[value] = save_var
        best_res_df = best_res_df.append({'param': value,
                                          'value': params[value],
                                          'min_mean_val_loss': min_mean_loss},
                                         ignore_index=True)
        best_res_df.to_csv(best_result_path, header=False, index=False,
                           mode='a')
        best_res_df = pd.DataFrame(columns=csv_best_res)
else:
    x_train, x_test, y_train, y_test = load_new_data()
    for i in range(training_number):
        train_model()
