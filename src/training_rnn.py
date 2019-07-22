import tensorflow
import pandas as pd
import numpy as np
import os
from time import time
from keras.layers import Dense, Dropout, CuDNNLSTM, CuDNNGRU, RNN, LSTM, GRU
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint

from kdd_processing import kdd_encoding
from unsw_processing import unsw_encoding
from results_visualisation import print_results

# Allows tensorflow to run multiple sessions (Multiple learning simultaneously)
# Comment the 3 following lines if causing issues
# config = tensorflow.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tensorflow.Session(config=config)

csv_values = ['epochs', 'acc', 'loss', 'val_acc', 'val_loss', "train_data",
              "features_nb", 'loss_fct', 'optimizer', 'activation_fct',
              'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type',
              'encoder']

csv_best_res = ['param', 'value', 'min_mean_val_loss']

# epochs: Number of iteration of the training dataset
# train_data: Number of rows in training dataset (see processing files)
# features_nb: Number of features kept as input (see processing files)
# loss fct: Loss function used in training
# optimizer: Optimizer used in training
# activation_fct: Activation function used in outputs layer
# layer_nb: Number of hidden layers in the network
# unit_nb: Number of cells for each layer
# batch_size: Number of elements observed before updating weights
# dropout: Fraction of inputs randomly discarded
# cell_type: Type of cell ['CuDNNLSTM', 'CuDNNGRU', 'RNN', 'LSTM', 'GRU']
# encoder: Encoding performed (see processing files)
# dataset: Processing file to be called ['kdd', 'unsw']
# training_nb: Number of model to be trained with the same params
# resultstocsv: Wether to save results to csv
# resultstologs: Wether to save models and tensorboard logs
# showresults: Wether to show detailled statistics about the trained model
# shuffle: Wether to shuffle the batches sequences during training

# ***** REFERENCES PARAMETERS *****
params = {'epochs': 3, 'train_data': 494021, 'features_nb': 4,
          'loss_fct': 'mse', 'optimizer': 'rmsprop',
          'activation_fct': 'sigmoid', 'layer_nb': 1, 'unit_nb': 128,
          'batch_size': 1024, 'dropout': 0.2, 'cell_type': 'CuDNNLSTM',
          'encoder': 'labelencoder', 'dataset': 'kdd', 'training_nb': 1,
          'resultstocsv': False, 'resultstologs': False, 'showresults': True,
          'shuffle': True}

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
              # 'cell_type': ['CuDNNLSTM', 'CuDNNGRU', 'RNN', 'LSTM', 'GRU'],
              # 'dataset : ['kdd', 'unsw']
              }

model_path = './models/'
logs_path = './logs/'
res_path = './results/' + 'testcsv/'

if params['resultstologs'] is True:
    res_name = str(params['train_data']) + '_' + str(params['features_nb']) +\
        '_' + params['loss_fct'] + '_' + params['optimizer'] + '_' +\
        params['activation_fct'] + '_' + str(params['layer_nb']) + '_' +\
        str(params['unit_nb']) + '_' + str(params['batch_size']) + '_' +\
        str(params['dropout']) + '_' + params['cell_type'] + '_' +\
        params['encoder'] + '_' + str(time())


# Encode dataset and return : x_train, x_test, y_train, y_tests
def load_data():
    if params['dataset'] == 'kdd':
        x_train, x_test, y_train, y_test = kdd_encoding(params)
    elif params['dataset'] == 'unsw':
        x_train, x_test, y_train, y_test = unsw_encoding(params)

    # Reshape the inputs in the accepted model format
    x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
    x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
    return x_train, x_test, y_train, y_test


# Create and train a model
def train_model(x_train, x_test, y_train, y_test):
    if params['cell_type'] == 'CuDNNLSTM':
        cell = CuDNNLSTM
    elif params['cell_type'] == 'CuDNNGRU':
        cell = CuDNNGRU
    elif params['cell_type'] == 'RNN':
        cell = RNN
    elif params['cell_type'] == 'LSTM':
        cell = LSTM
    elif params['cell_type'] == 'GRU':
        cell = GRU

    # Create a Sequential layer, one layer after the other
    model = Sequential()
    # If there is more than 1 layer, the first must return sequences
    for _ in range(params['layer_nb']-1):
        model.add(cell(units=params['unit_nb'],
                       input_shape=(x_train.shape[1:]), return_sequences=True))
        model.add(Dropout(rate=params['dropout']))

    # If there is only 1 layer, it must not return sequences
    if(params['layer_nb'] == 1):
        model.add(cell(units=params['unit_nb'], input_shape=x_train.shape[1:]))
        model.add(Dropout(rate=params['dropout']))
    else:  # If there is more than 1, the following must not return sequences
        model.add(cell(units=params['unit_nb']))
        model.add(Dropout(rate=params['dropout']))
    # Outputs layer
    model.add(Dense(units=y_train.shape[1],
                    activation=params['activation_fct']))

    model.compile(loss=params['loss_fct'], optimizer=params['optimizer'],
                  metrics=['accuracy'])

    # Create model and logs folder if does not exist
    if params['resultstologs'] is True:
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        save_model = ModelCheckpoint(filepath=model_path + res_name,
                                     monitor='val_acc', save_best_only=True)
        tensorboard = TensorBoard(logs_path+res_name)
        callbacks = [save_model, tensorboard]
    else:
        callbacks = None

    model.summary()

    hist = model.fit(x_train, y_train, params['batch_size'], params['epochs'],
                     verbose=1, shuffle=params['shuffle'],
                     validation_data=(x_test, y_test), callbacks=callbacks)

    if params['showresults'] is True:
        print_results(params, model, x_train, x_test, y_train, y_test)

    return hist


def res_to_csv():
    ref_min_val_loss = 10  # Minimal reference loss value
    nsmall = 5  # Number of val loss for the mean val loss

    # Create the results directory if it doesnt exist
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    full_res_path = res_path + 'full_results.csv'
    best_res_path = res_path + 'best_result.csv'

    # Initialize results and best_results dataframes
    results_df = pd.DataFrame(columns=csv_values)
    results_df.to_csv(full_res_path, index=False)

    best_res_df = pd.DataFrame(columns=csv_best_res)

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

    # Make the mean of the n smallest val_loss for each feature values
    def min_mean_val_loss(feature):
        # Load the results previously saved as csv for the features
        df = pd.read_csv(res_path+feature+".csv", index_col=False)
        names = df[feature].unique().tolist()
        df_loss = pd.DataFrame(columns=names)

        # For each value of the feature, compare the n smallest val loss
        for i in range(len(names)):
            df_value_loss = df.loc[df[feature] == names[i]]
            df_value_loss = df_value_loss.nsmallest(nsmall, 'val_loss')
            df_loss[names[i]] = np.array(df_value_loss['val_loss'])

        # Return the index and the value of the feature
        #  with the smallest mean val loss
        return df_loss.mean().idxmin(), df_loss.mean().min()

    for feature in params_var.keys():
        results_df.to_csv(res_path + feature + ".csv", index=False)
        save_feature_value = params[feature]

        for feature_value in params_var[feature]:
            df_value = pd.DataFrame(columns=csv_values)
            params[feature] = feature_value

            if feature == 'encoder' or feature == 'train_data':
                # The encoding will have to change, so the data are reaload
                x_train, x_test, y_train, y_test = load_data()

            for _ in range(params['training_nb']):
                history = train_model(x_train, x_test, y_train, y_test)

                # The datafranme is filled for each epoch
                for epoch in range(params['epochs']):
                    df_value = fill_dataframe(df_value, history, epoch)
            # At the end of the training, results are saved in csv
            df_value.to_csv(full_res_path, header=False, index=False, mode='a')
            df_value.to_csv(res_path + feature + ".csv", header=False,
                            index=False, mode='a')
        # Once the test of the value is over, return the min mean val loss
        feature_value_min_loss, min_mean_loss = min_mean_val_loss(feature)

        # Compare the best val loss for the feature value with the reference
        # of the best val loss. If better, the best val becomes the reference,
        # and feature value correspondind is chosen for the rest of the test
        if min_mean_loss < ref_min_val_loss:
            params[feature] = feature_value_min_loss
            ref_min_val_loss = min_mean_loss
        else:
            params[feature] = save_feature_value

        # Save the best feature value, reference is saved if better
        best_res_df = best_res_df.append({'param': feature,
                                          'value': params[feature],
                                          'min_mean_val_loss': min_mean_loss},
                                         ignore_index=True)
        best_res_df.to_csv(best_res_path, index=False)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()

    for i in range(params['training_nb']):
        if params['resultstocsv'] is False:
            train_model(x_train, x_test, y_train, y_test)
        else:
            res_to_csv()
