from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
from tensorflow._api.v1.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,
                                   LabelEncoder, MinMaxScaler)
from sklearn.metrics import (confusion_matrix, roc_auc_score, precision_score,
                             roc_curve, recall_score, auc)
pd.options.mode.chained_assignment = None

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

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
                 "dst_host_srv_rerror_rate", "label", "difficuly"]

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
params = {'epochs': 100, 'train_data': 25191, 'features_nb': 41,
          'batch_size': 1024, 'encoder': 'standarscaler'}

# ***** VARIABLE PARAMETERS *****
params_var = {'encoder': ['standardscaler', 'labelencoder',
                          'minmaxscaler01', 'minmaxscaler11',
                          'ordinalencoder'],
              'batch_size': [128, 256, 512, 1024, 2048],
              # 'features_nb': [4, 8, 41],
              # 'train_data': [494021, 4898431, 125973, 25191],
              # 'cell_type': ['CuDNNLSTM', 'CuDNNGRU', 'SimpleRNN'],
              }


def data_processing():
    if params['train_data'] == 494021:
        train_data_path = "./data/kddcup.traindata_10_percent_corrected.csv"
        test_data_path = "./data/kddcup.testdata_10_percent_corrected.csv"
    elif params['train_data'] == 4898431:
        train_data_path = "./data/kddcup.traindata.corrected.csv"
        test_data_path = "./data/kddcup.testdata_10_percent_corrected.csv"
    elif params['train_data'] == 125973:
        train_data_path = "./data/KDDTrain+.csv"
        test_data_path = "./data/KDDTest+.csv"
    elif params['train_data'] == 25191:
        train_data_path = "./data/KDDTrain+_20Percent.csv"
        test_data_path = "./data/KDDTest-21.csv"

    train_dataframe = pd.read_csv(train_data_path, names=full_features)
    test_dataframe = pd.read_csv(test_data_path, names=full_features)

    def process_dataframe(dataframe):
        if params['features_nb'] == 4:
            features = four_features
        elif ['features_nb'] == 8:
            features = eight_features
        else:
            features = full_features

        dataframe = dataframe[features]

        dataframe['label'] = dataframe['label'].replace('normal.', 0)
        dataframe['label'] = dataframe['label'].replace('normal', 0)
        for i in range(len(probe)):
            dataframe['label'] = dataframe['label'].replace(probe[i], 1)
            dataframe['label'] = dataframe['label'].replace(probe[i][:-1], 1)
        for i in range(len(dos)):
            dataframe['label'] = dataframe['label'].replace(dos[i], 2)
            dataframe['label'] = dataframe['label'].replace(dos[i][:-1], 2)
        for i in range(len(u2r)):
            dataframe['label'] = dataframe['label'].replace(u2r[i], 3)
            dataframe['label'] = dataframe['label'].replace(u2r[i][:-1], 3)
        for i in range(len(r2l)):
            dataframe['label'] = dataframe['label'].replace(r2l[i], 4)
            dataframe['label'] = dataframe['label'].replace(r2l[i][:-1], 4)

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

    x_train, Y_train = process_dataframe(train_dataframe)
    x_test, Y_test = process_dataframe(test_dataframe)

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


x_train, x_test, y_train, y_test = data_processing()

model = load_model('./models/1562057554.3048038')
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
