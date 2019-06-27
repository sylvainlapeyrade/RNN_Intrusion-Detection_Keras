from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, CuDNNGRU
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder, MinMaxScaler
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

results = pd.DataFrame(columns=csv_values)
results.to_csv('./dropout.csv', index=False)

# Variables des chemins
train_data_name = "kddcup.traindata_10_percent_corrected"
# train_data_name = "kddcup.traindata.corrected"
test_data_name = "kddcup.testdata_10_percent_corrected"
train_data_path = "./data/kddcup.traindata_10_percent_corrected"
# train_data_path = "./data/kddcup.traindata.corrected"
test_data_path = "./data/kddcup.testdata_10_percent_corrected"

# Parameters
# features_number = eight_features
# features_number = full_features
# features_values = [four_features, eight_features, full_features]
# dropout_rate = 0.2
dropout_rate_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
optimizer = 'rmsprop'
# optimizer_values = ['adam', 'sgd', 'rmsprop', 'nadam', 'adamax', 'adadelta']
batch_size = 1024
# batch_size_values = [512, 1024, 2048]
activation_function = 'sigmoid'
# activation_function_values = ['sigmoid', 'softmax', 'relu', 'tanh']
layer_number = 1
# layer_number_values = [1, 2, 3, 4]
unit_number = 128
# unit_number_values = [4, 8, 32, 64, 128]
training_number = 10
epochs = 100
loss_function = 'mse'
cell_type = CuDNNLSTM
# cell_type_values = [CuDNNLSTM, CuDNNGRU]
encoder = 'standardscaler'
# encoder_values = ['standardscaler', 'labelencoder', 'minmaxscaler01',
# 'minmaxscaler11', 'ordinalencoder']

# for encoder in encoder_values:
train_dataframe = pd.read_csv(train_data_path, names=full_features)
test_dataframe = pd.read_csv(test_data_path, names=full_features)


def process_dataframe(dataframe, name, features_number):
    dataframe = dataframe[features_number]

    total_data_length = len(dataframe)
    normal_data_length = len(dataframe[dataframe['label'] == 'normal.'])
    anormal_data_length = len(dataframe[dataframe['label'] != 'normal.'])

    def print_data(attack_name, attack_lengh):
        print(' '*6 + attack_name + ': {:,} soit {}%'
              .format(attack_lengh,
                      round(attack_lengh * 100 / total_data_length, 3))
              .replace(',', ' '))

    dataframe['label'] = dataframe['label'].replace('normal.', 1)
    for i in range(len(probe)):
        dataframe['label'] = dataframe['label'].replace(probe[i], 1)
    for i in range(len(dos)):
        dataframe['label'] = dataframe['label'].replace(dos[i], 2)
    for i in range(len(u2r)):
        dataframe['label'] = dataframe['label'].replace(u2r[i], 3)
    for i in range(len(r2l)):
        dataframe['label'] = dataframe['label'].replace(r2l[i], 4)

    x = dataframe[features_number[:-1]]
    y = dataframe['label']

    if encoder == "standardscaler":
        if 'service' in features_number:
            for i in range(len(service_values)):
                x['service'] = x['service'].replace(service_values[i], i)

        if 'protocol_type' in features_number:
            for i in range(len(protocol_type_values)):
                x['protocol_type'] = x['protocol_type'].replace(
                    protocol_type_values[i], i)

        if 'flag' in features_number:
            for i in range(len(flag_values)):
                x['flag'] = x['flag'].replace(flag_values[i], i)

        x = StandardScaler().fit_transform(x)
    elif encoder == 'labelencoder':
        x = np.array(x.apply(LabelEncoder().fit_transform))
    elif encoder == 'minmaxscaler01':
        if 'service' in features_number:
            for i in range(len(service_values)):
                x['service'] = x['service'].replace(service_values[i], i)

        if 'protocol_type' in features_number:
            for i in range(len(protocol_type_values)):
                x['protocol_type'] = x['protocol_type'].replace(
                    protocol_type_values[i], i)

        if 'flag' in features_number:
            for i in range(len(flag_values)):
                x['flag'] = x['flag'].replace(flag_values[i], i)

        x = np.array(MinMaxScaler(feature_range=(0, 1)).fit_transform(x))
    elif encoder == 'minmaxscaler11':
        if 'service' in features_number:
            for i in range(len(service_values)):
                x['service'] = x['service'].replace(service_values[i], i)

        if 'protocol_type' in features_number:
            for i in range(len(protocol_type_values)):
                x['protocol_type'] = x['protocol_type'].replace(
                    protocol_type_values[i], i)

        if 'flag' in features_number:
            for i in range(len(flag_values)):
                x['flag'] = x['flag'].replace(flag_values[i], i)

        x = np.array(MinMaxScaler(feature_range=(-1, 1)).fit_transform(x))
    elif encoder == 'ordinalencoder':
        x = np.array(OrdinalEncoder().fit_transform(x))
    x = np.array(x)
    return x.reshape([-1, x.shape[1], 1]), y


x_train, Y_train = process_dataframe(
    train_dataframe, 'entrainement', four_features)
x_test, Y_test = process_dataframe(test_dataframe, 'test', four_features)


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

if x_train.shape[0] == 4898431:
    train_data = 'full'
elif x_train.shape[0] == 494021:
    train_data = '10p'

if cell_type == CuDNNLSTM:
    cell = 'CuDNNLSTM'
elif cell_type == CuDNNGRU:
    cell = 'CuDNNGRU'


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

    model.add(
        Dense(units=y_train.shape[1], activation=activation_function))

    model.compile(loss=loss_function, optimizer=optimizer,
                  metrics=['accuracy'])

    return model.fit(x_train, y_train, epochs=epochs, shuffle=True,
                     batch_size=batch_size, verbose=2,
                     validation_data=(x_test, y_test),
                     use_multiprocessing=True)


for dropout_rate in dropout_rate_values:
    # for optimizer in optimizer_values:
    # for batch_size in batch_size_values:
    # for activation_function in activation_function_values:
    # for layer_number in layer_number_values:
    # for unit_number in unit_number_values:
    for i in range(training_number):
        history = train_model()
        for j in range(epochs):
            results = results.append({'epochs': j,
                                      'acc':  history.history['acc'][j],
                                      'loss': history.history['loss'][j],
                                      'val_acc': history.history['val_acc'][j],
                                      'val_loss': history.history['val_loss'][j],
                                      'train_data': train_data,
                                      'features_nb': x_train.shape[1],
                                      'loss_fct': loss_function,
                                      'optimizer': optimizer,
                                      'activation_fct': activation_function,
                                      'layer_nb': layer_number,
                                      'unit_nb': unit_number,
                                      'batch_size': batch_size,
                                      'dropout': dropout_rate,
                                      'cell_type': cell,
                                      'encoder': encoder},
                                     ignore_index=True)
    results.to_csv('./dropout.csv', header=False, index=False, mode='a')
    results = pd.DataFrame(columns=csv_values)
