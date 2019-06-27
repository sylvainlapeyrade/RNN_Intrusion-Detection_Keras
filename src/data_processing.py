import configparser as cp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None

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

# Chargement des datasets
config_parser = cp.ConfigParser()
config_parser.read('./config.ini')
print('Chargement du dataset d\'entrainement : {}...'
      .format(config_parser.get('DataPath', 'train_data_name')))
print('Chargement du dataset de test : {}...'
      .format(config_parser.get('DataPath', 'test_data_name')))

# On parse le dataset en format csv avec Pandas et attribut à chaque colonne
# le nom de la feature, on obtient alors un dataframe (equivaut à une matrice)
train_dataframe = pd.read_csv(config_parser.get(
    'DataPath', 'train_data_path'), names=full_features)
test_dataframe = pd.read_csv(config_parser.get(
    'DataPath', 'test_data_path'), names=full_features)


def process_dataframe(dataframe, name, features_number):
    # Réduit le dataframe en ne conservant que les features "utiles"
    dataframe = dataframe[features_number]

    # Récupère le nombre de chaque type de données pour afficher des stats
    total_data_length = len(dataframe)
    normal_data_length = len(dataframe[dataframe['label'] == 'normal.'])
    anormal_data_length = len(dataframe[dataframe['label'] != 'normal.'])

    def print_data(attack_name, attack_lengh):
        print(' '*6 + attack_name + ': {:,} soit {}%'
              .format(attack_lengh,
                      round(attack_lengh * 100 / total_data_length, 3))
              .replace(',', ' '))

    print('\nJeu de données {}:'.format(name))
    print('Total Entrées: {:,}'.format(total_data_length).replace(',', ' '))
    print_data("Entrées normales", normal_data_length)
    print_data("Entrées anormales", anormal_data_length)

    # Assigne numero différent selon la nature de la connexion
    dataframe['label'] = dataframe['label'].replace('normal.', 1)
    for i in range(len(probe)):
        dataframe['label'] = dataframe['label'].replace(probe[i], 1)
    for i in range(len(dos)):
        dataframe['label'] = dataframe['label'].replace(dos[i], 2)
    for i in range(len(u2r)):
        dataframe['label'] = dataframe['label'].replace(u2r[i], 3)
    for i in range(len(r2l)):
        dataframe['label'] = dataframe['label'].replace(r2l[i], 4)

    # Récupère le nombre de chaque attaque pour afficher des stats
    probe_data_length = len(dataframe[dataframe['label'] == 1])
    dos_data_length = len(dataframe[dataframe['label'] == 2])
    u2r_data_length = len(dataframe[dataframe['label'] == 3])
    r2l_data_length = len(dataframe[dataframe['label'] == 4])

    print("Dont:")
    print_data("Probe", probe_data_length)
    print_data("DoS", dos_data_length)
    print_data("U2R", u2r_data_length)
    print_data("R2L", r2l_data_length)

    # Création des tableaux d'entrées X et de sortie y
    x = dataframe[features_number[:-1]]
    y = dataframe['label']

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

    # Standardise en centrant les données sur la moyenne et l'écart type
    x = StandardScaler().fit_transform(x)

    return np.array(x), y


x_train, Y_train = process_dataframe(
    train_dataframe, 'entrainement', four_features)
x_test, Y_test = process_dataframe(test_dataframe, 'test', four_features)


def oneHotEncoding(y_label_encoded):
    # Encodage one-hot de y (=> valeur 0 ou 1)
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

np.save(config_parser.get('DataPath', 'x_train_path'),
        x_train.reshape([-1, x_train.shape[1], 1]))
np.save(config_parser.get('DataPath', 'y_train_path'), y_train)
np.save(config_parser.get('DataPath', 'x_test_path'),
        x_test.reshape([-1, x_test.shape[1], 1]))
np.save(config_parser.get('DataPath', 'y_test_path'), y_test)

print("\nEnregistrement des datasets fini.")
