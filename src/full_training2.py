from tensorflow.keras.layers import (Dense, Dropout, CuDNNLSTM, CuDNNGRU,
                                     SimpleRNN)
from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,
                                   LabelEncoder, MinMaxScaler)
pd.options.mode.chained_assignment = None

full_features = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                 "num_failed_logins", "logged_in", "num_compromised",
                 "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files",
                 "num_outbound_cmds", "is_host_login", "is_guest_login",
                 "count", "srv_count", "serror_rate", "srv_serror_rate",
                 "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                 "dst_host_srv_count", "dst_host_same_srv_rate",
                 "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                 "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                 "dst_host_srv_rerror_rate", "label"]

four_features = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                 'dst_host_rerror_rate', 'label']

eight_features = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                  'dst_host_rerror_rate', 'dst_bytes', 'hot',
                  'num_failed_logins', 'dst_host_srv_count', 'label']

probe = ['ipsweep.', 'nmap.', 'portsweep.', 'satan.', 'saint.', 'mscan.']
dos = ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.', 'apache2.',
       'udpstorm.', 'processtable.', 'mailbomb.']
u2r = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.', 'xterm.', 'ps.',
       'sqlattack.']
r2l = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.',
       'warezclient.', 'warezmaster.', 'snmpgetattack.', 'named.', 'xlock.',
       'xsnoop.', 'sendmail.', 'httptunnel.', 'worm.', 'snmpguess.']

service_values = ['http', 'smtp', 'finger', 'domain_u', 'auth', 'telnet',
                  'ftp', 'eco_i', 'ntp_u', 'ecr_i', 'other', 'private',
                  'pop_3', 'ftp_data', 'rje', 'time', 'mtp', 'link',
                  'remote_job', 'gopher', 'ssh', 'name', 'whois', 'domain',
                  'login', 'imap4', 'daytime', 'ctf', 'nntp', 'shell', 'IRC',
                  'nnsp', 'http_443', 'exec', 'printer', 'efs', 'courier',
                  'uucp', 'klogin', 'kshell', 'echo', 'discard', 'systat',
                  'supdup', 'iso_tsap', 'hostnames', 'csnet_ns', 'pop_2',
                  'sunrpc', 'uucp_path', 'netbios_ns', 'netbios_ssn',
                  'netbios_dgm', 'sql_net', 'vmnet', 'bgp', 'Z39_50', 'ldap',
                  'netstat', 'urh_i', 'X11', 'urp_i', 'pm_dump', 'tftp_u',
                  'tim_i', 'red_i', 'icmp', 'http_2784', 'harvest', 'aol',
                  'http_8001']

flag_values = ['OTH', 'RSTOS0', 'SF', 'SH',
               'RSTO', 'S2', 'S1', 'REJ', 'S3', 'RSTR', 'S0']

protocol_type_values = ['tcp', 'udp', 'icmp']

csv_values = ['epochs', 'acc', 'loss', 'val_acc', 'val_loss', "train_data",
              "features_nb", 'loss_fct', 'optimizer', 'activation_fct',
              'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type',
              'encoder']
csv_best_res = ['param', 'value', 'min_mean_val_loss']

# ***** REFERENCES PARAMETERS *****
params = {'epochs': 100, 'train_data': 494021, 'features_nb': 4,
          'loss_fct': 'mse', 'optimizer': 'nadam',
          'activation_fct': 'tanh', 'layer_nb': 2, 'unit_nb': 32,
          'batch_size': 512, 'dropout': 0.2, 'cell_type': 'CuDNNGRU',
          'encoder': 'standarscaler'}

# ***** VARIABLE PARAMETERS *****
params_var = {'encoder': ['standardscaler', 'labelencoder',
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'optimizer': ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax',
                            'adadelta'],
              'activation_fct': ['sigmoid', 'softmax', 'relu', 'tanh'],
              'layer_nb': [1, 2],
              'unit_nb': [4, 8, 32, 64, 128, 256],
              'dropout': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
              'batch_size': [128, 256, 512, 1024, 2048],
              # 'features_nb': [4, 8, 41],
              # 'train_data': [494021, 4898431],
              # 'cell_type': ['CuDNNLSTM', 'CuDNNGRU', 'SimpleRNN'],
              }

training_number = 10
# min_val_loss = 0.024801729255749035
min_val_loss = 0.040801729255749035

# ***** PATH *****
test_data_path = "./data/kddcup.testdata_10_percent_corrected"
results_path = "./full_results.csv"
best_result_path = "./best_result.csv"
results_df = pd.DataFrame(columns=csv_values)
results_df.to_csv(results_path, index=False)
best_res_df = pd.DataFrame(columns=csv_best_res)
best_res_df.to_csv(best_result_path, index=False)


def data_processing():
    if params['train_data'] == 494021:
        train_data_path = "./data/kddcup.traindata_10_percent_corrected"
    elif params['train_data'] == 4898431:
        train_data_path = "./data/kddcup.traindata.corrected"
    train_dataframe = pd.read_csv(train_data_path, names=full_features)
    test_dataframe = pd.read_csv(test_data_path, names=full_features)

    def process_dataframe(dataframe, features_number):
        if features_number == 4:
            features = four_features
        elif features_number == 8:
            features = eight_features
        else:
            features = full_features

        dataframe = dataframe[features]

        dataframe['label'] = dataframe['label'].replace('normal.', 1)
        for i in range(len(probe)):
            dataframe['label'] = dataframe['label'].replace(probe[i], 1)
        for i in range(len(dos)):
            dataframe['label'] = dataframe['label'].replace(dos[i], 2)
        for i in range(len(u2r)):
            dataframe['label'] = dataframe['label'].replace(u2r[i], 3)
        for i in range(len(r2l)):
            dataframe['label'] = dataframe['label'].replace(r2l[i], 4)

        x = dataframe[features[:-1]]
        y = dataframe['label']

        if params['encoder'] == 'ordinalencoder':
            x = np.array(OrdinalEncoder().fit_transform(x))
        elif params['encoder'] == 'labelencoder':
            x = np.array(x.apply(LabelEncoder().fit_transform))
        else:
            if 'service' in features:
                for i in range(len(service_values)):
                    x['service'] = x['service'].replace(service_values[i], i)

            if 'protocol_type' in features:
                for i in range(len(protocol_type_values)):
                    x['protocol_type'] = x['protocol_type'].replace(
                        protocol_type_values[i], i)

            if 'flag' in features:
                for i in range(len(flag_values)):
                    x['flag'] = x['flag'].replace(flag_values[i], i)
            if params['encoder'] == "standardscaler":
                x = StandardScaler().fit_transform(x)
            elif params['encoder'] == "minmaxscaler01":
                x = np.array(MinMaxScaler(
                    feature_range=(0, 1)).fit_transform(x))
            elif params['encoder'] == "minmaxscaler11":
                x = np.array(MinMaxScaler(
                    feature_range=(-1, 1)).fit_transform(x))
        x = np.array(x)
        return x.reshape([-1, x.shape[1], 1]), y

    x_train, Y_train = process_dataframe(
        train_dataframe, params['features_nb'])
    x_test, Y_test = process_dataframe(test_dataframe, params['features_nb'])

    def oneHotEncoding(y_label_encoded):
        y_one_hot = np.zeros([y_label_encoded.shape[0], 5])
        for i in range(y_label_encoded.shape[0]):
            if y_label_encoded[i] == 0:
                y_one_hot[i, 0] = 1
            elif y_label_encoded[i] == 1:
                y_one_hot[i, 1] = 1
            elif y_label_encoded[i] == 2:
                y_one_hot[i, 2] = 1
            elif y_label_encoded[i] == 3:
                y_one_hot[i, 3] = 1
            elif y_label_encoded[i] == 4:
                y_one_hot[i, 4] = 1
        return y_one_hot

    y_train = oneHotEncoding(Y_train)
    y_test = oneHotEncoding(Y_test)

    return x_train, x_test, y_train, y_test


def train_model():
    if params['cell_type'] == 'CuDNNLSTM':
        cell = CuDNNLSTM
    elif params['cell_type'] == 'CuDNNGRU':
        cell = CuDNNGRU
    elif params['cell_type'] == 'simpleRNN':
        cell = CuDNNGRU

    model = Sequential()
    for _ in range(params['layer_nb']-1):
        model.add(cell(units=params['unit_nb'], input_shape=(
            x_train.shape[1:]), return_sequences=True))
        model.add(Dropout(rate=params['dropout']))

    if(params['layer_nb'] == 1):
        model.add(cell(units=params['unit_nb'], input_shape=(
            x_train.shape[1:])))
        model.add(Dropout(rate=params['dropout']))
    else:
        model.add(cell(units=params['unit_nb']))
        model.add(Dropout(rate=params['dropout']))

    model.add(Dense(units=y_train.shape[1],
                    activation=params['activation_fct']))

    model.compile(loss=params['loss_fct'], optimizer=params['optimizer'],
                  metrics=['accuracy'])

    return model.fit(x_train, y_train, epochs=params['epochs'], shuffle=True,
                     batch_size=params['batch_size'], verbose=2,
                     validation_data=(x_test, y_test),
                     use_multiprocessing=True)


def fill_dataframe(df, history, epoch):
    return df.append({'epochs': epoch,
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


def get_min_mean_value(value):
    df = pd.read_csv("./"+value+".csv", index_col=False)
    names = df[value].unique().tolist()
    df_loss = pd.DataFrame(columns=names)

    for i in range(len(names)):
        df_value_loss = df.loc[df[value] == names[i]]
        df_value_loss = df_value_loss.nsmallest(5, 'val_loss')
        df_loss[names[i]] = np.array(df_value_loss['val_loss'])

    return df_loss.mean().idxmin(), df_loss.mean().min()


for value in params_var.keys():
    x_train, x_test, y_train, y_test = data_processing()
    results_df.to_csv("./" + value + ".csv", index=False)
    save_var = params[value]

    for var in params_var[value]:
        value_df = pd.DataFrame(columns=csv_values)
        params[value] = var
        if value == 'encoder' or value == 'train_data':
            x_train, x_test, y_train, y_test = data_processing()

        for _ in range(training_number):
            history = train_model()
            for epoch in range(params['epochs']):
                value_df = fill_dataframe(value_df, history, epoch)

        value_df.to_csv(results_path, header=False, index=False, mode='a')
        value_df.to_csv("./"+value+".csv", header=False, index=False, mode='a')
    param_min_value, min_mean_loss = get_min_mean_value(value)

    if min_mean_loss < min_val_loss:
        params[value] = param_min_value
        min_val_loss = min_mean_loss
    else:
        params[value] = save_var

    best_res_df = pd.DataFrame([{'param': value, 'value': params[value],
                                 'min_mean_val_loss': min_mean_loss}])
    best_res_df.to_csv(best_result_path, header=False, index=False, mode='a')
    best_res_df.iloc[0:0]  # Drop the df content efficiently
