from tensorflow._api.v1.keras.layers import Dense, Dropout, CuDNNLSTM, CuDNNGRU
from tensorflow._api.v1.keras.models import Sequential
from tensorflow._api.v1.keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import numpy as np
import configparser as cp
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore')

# Nom et description des features: kdd.ics.uci.edu/databases/kddcup99/task.html
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

# On ne garde que quelques features utiles en accord avec l'article: "Applying
#  long short-term memory recurrent neural networks to intrusion detection"
# Les 4 features les plus importantes + label
four_features = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                 'dst_host_rerror_rate', 'label']

# Les 8 feature les plus importantes + label
eight_features = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                  'dst_host_rerror_rate', 'dst_bytes', 'hot',
                  'num_failed_logins', 'dst_host_srv_count', 'label']

# Type d'attaque: kdd.ics.uci.edu/databases/kddcup99/training_attack_types
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
              'layer_nb', 'unit_nb', 'batch_size', 'dropout', 'cell_type']


# Variables des chemins
train_data_name = "kddcup.traindata_10_percent_corrected"
# train_data_name = "kddcup.traindata.corrected"
test_data_name = "kddcup.testdata_10_percent_corrected"
train_data_path = "./data/kddcup.traindata_10_percent_corrected"
# train_data_path = "./data/kddcup.traindata.corrected"
test_data_path = "./data/kddcup.testdata_10_percent_corrected"
model_path = "./models/"
log_path = "./logs/"

config_parser = cp.ConfigParser()
config_parser.read('./config.ini')

x_train = np.load(config_parser.get('DataPath', 'x_train_path'))
y_train = np.load(config_parser.get('DataPath', 'y_train_path'))
x_test = np.load(config_parser.get('DataPath', 'x_test_path'))
y_test = np.load(config_parser.get('DataPath', 'y_test_path'))

# Parameters
# features = four_features
# features_number = eight_features
# features_number = full_features
# features_values = [four_features, eight_features, full_features]
dropout_rate = 0.2
# dropout_rate_values = [0.1, 0.2, 0.3]
optimizer = 'adam'
# optimizer_values = ['adam', 'sgd', 'rmsprop']
batch_size = 1024
# batch_size_values = [512, 1024, 2048]
activation_function = 'sigmoid'
# activation_function_values = ['sigmoid', 'softmax', 'relu', 'tanh']
layer_number = 2
# layer_number_values = [1, 2, 3]
unit_number = 128
# unit_number_values = [64, 128, 256]
training_number = 2
epochs = 2
loss_function = 'mse'
cell_type = CuDNNLSTM
cell_type_values = [CuDNNLSTM, CuDNNLSTM]

# # Chargement des datasets
# print('Chargement du dataset d\'entrainement : {}...'.format(train_data_name))
# print('Chargement du dataset de test : {}...'.format(test_data_name))

# # On parse le dataset en format csv avec Pandas et attribut à chaque colonne
# # le nom de la feature, on obtient alors un dataframe (equivaut à une matrice)
# train_dataframe = pd.read_csv(train_data_path, names=full_features)
# test_dataframe = pd.read_csv(test_data_path, names=full_features)


# def process_dataframe(dataframe, name, features_number):
#     # Réduit le dataframe en ne conservant que les features "utiles"
#     dataframe = dataframe[features_number]

#     # Récupère le nombre de chaque type de données pour afficher des stats
#     total_data_length = len(dataframe)
#     normal_data_length = len(dataframe[dataframe['label'] == 'normal.'])
#     anormal_data_length = len(dataframe[dataframe['label'] != 'normal.'])

#     def print_data(attack_name, attack_lengh):
#         print(' '*6 + attack_name + ': {:,} soit {}%'
#               .format(attack_lengh,
#                       round(attack_lengh * 100 / total_data_length, 3))
#               .replace(',', ' '))

#     print('\nJeu de données {}:'.format(name))
#     print('Total Entrées: {:,}'.format(total_data_length).replace(',', ' '))
#     print_data("Entrées normales", normal_data_length)
#     print_data("Entrées anormales", anormal_data_length)

#     # Assigne numero différent selon la nature de la connexion
#     dataframe.loc[dataframe['label'] == 'normal.', 'label'] = 0
#     for i in range(len(probe)):
#         dataframe.loc[dataframe['label'] == probe[i], 'label'] = 1
#     for i in range(len(dos)):
#         dataframe.loc[dataframe['label'] == dos[i], 'label'] = 2
#     for i in range(len(u2r)):
#         dataframe.loc[dataframe['label'] == u2r[i], 'label'] = 3
#     for i in range(len(r2l)):
#         dataframe.loc[dataframe['label'] == r2l[i], 'label'] = 4

#     # Récupère le nombre de chaque attaque pour afficher des stats
#     probe_data_length = len(dataframe[dataframe['label'] == 1])
#     dos_data_length = len(dataframe[dataframe['label'] == 2])
#     u2r_data_length = len(dataframe[dataframe['label'] == 3])
#     r2l_data_length = len(dataframe[dataframe['label'] == 4])

#     print("Dont:")
#     print_data("Probe", probe_data_length)
#     print_data("DoS", dos_data_length)
#     print_data("U2R", u2r_data_length)
#     print_data("R2L", r2l_data_length)

#     # Création des tableaux d'entrées X et de sortie y
#     x = dataframe[features_number[:-1]]
#     y = dataframe['label']

#     if 'service' in features_number:
#         for i in range(len(service_values)):
#             x.loc[x['service'] == service_values[i], 'service'] = i

#     if 'protocol_type' in features_number:
#         for i in range(len(protocol_type_values)):
#             x.loc[x['protocol_type'] ==
#                   protocol_type_values[i], 'protocol_type'] = i

#     if 'flag' in features_number:
#         for i in range(len(flag_values)):
#             x.loc[x['flag'] == flag_values[i], 'flag'] = i

#     # Standardise en centrant les données sur la moyenne et l'écart type
#     x = StandardScaler().fit_transform(x)

#     return x.reshape([-1, x.shape[1], 1]), y


# x_train, Y_train = process_dataframe(
#     train_dataframe, 'entrainement', four_features)
# x_test, Y_test = process_dataframe(test_dataframe, 'test', four_features)


# def oneHotEncoding(y_label_encoded):
#     # Encodage one-hot de y (=> valeur 0 ou 1)
#     y_one_hot = np.zeros([y_label_encoded.shape[0], 5])
#     for i in range(y_label_encoded.shape[0]):
#         if y_label_encoded[i] == 0:
#             y_one_hot[i, 0] = 1
#         elif y_label_encoded[i] == 1:
#             y_one_hot[i, 1] = 1
#         elif y_label_encoded[i] == 2:
#             y_one_hot[i, 2] = 1
#         elif y_label_encoded[i] == 3:
#             y_one_hot[i, 3] = 1
#         elif y_label_encoded[i] == 4:
#             y_one_hot[i, 4] = 1
#     return y_one_hot


# y_train = oneHotEncoding(Y_train)
# y_test = oneHotEncoding(Y_test)

# if train_data_name == "kddcup.traindata_10_percent_corrected":
#     train_data = '10p'
# else:
#     train_data = 'full'
if x_train.shape[0] == 4898431:
    train_data = 'full'
elif x_train.shape[0] == 494021:
    train_data = '10p'


def train_model():
    model = Sequential()
    for j in range(layer_number-1):
        model.add(cell_type(units=unit_number, input_shape=(
            x_train.shape[1:]), return_sequences=True))
        model.add(Dropout(rate=dropout_rate))

    if(layer_number == 1):
        model.add(cell_type(units=unit_number, input_shape=(
            x_train.shape[1:])))
        model.add(Dropout(rate=dropout_rate))
    else:
        model.add(cell_type(units=unit_number))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=y_train.shape[1], activation=activation_function))

    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=['accuracy'])

    return model.fit(x_train, y_train, epochs=epochs, shuffle=True,
                     batch_size=batch_size,
                     validation_data=(x_test, y_test),
                     use_multiprocessing=True)


results = pd.DataFrame(columns=csv_values)

# for dropout_rate in dropout_rate_values:
#     for optimizer in optimizer_values:
#         for batch_size in batch_size_values:
#             for activation_function in activation_function_values:
#                 for layer_number in layer_number_values:
#                     for unit_number in unit_number_values:
for i in range(training_number):
    history = train_model()
    for j in range(epochs):
        results = results.append({'epochs': j,
                                  'acc':  history.history['acc'][j],
                                  'loss': history.history['loss'][j],
                                  'val_acc': history.history['val_acc'][j],
                                  'val_loss':
                                  history.history['val_loss'][j],
                                  'train_data': train_data,
                                  'features_nb': x_train.shape[1],
                                  'loss_fct': loss_function,
                                  'optimizer': optimizer,
                                  'activation_fct': activation_function,
                                  'layer_nb': layer_number,
                                  'unit_nb': unit_number,
                                  'batch_size': batch_size,
                                  'dropout': dropout_rate,
                                  'cell_type': 'CuDNNLSTM'},
                                 ignore_index=True)

results.to_csv('./test.csv', index=False)
