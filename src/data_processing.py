import configparser as cp
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

# Nom et description des features: kdd.ics.uci.edu/databases/kddcup99/task.html
feature_name = ["duration", "protocol_type", "service", "flag", "src_bytes",
                "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                "num_failed_logins", "logged_in", "num_compromised",
                "root_shell", "su_attempted", "num_root", "num_file_creations",
                "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login", "count", "srv_count",
                "serror_rate", "srv_serror_rate", "rerror_rate",
                "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
                "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# On ne garde que quelques features utiles en accord avec l'article: "Applying
#  long short-term memory recurrent neural networks to intrusion detection"
# Les 4 premières feature sont les plus importantes avec 4 autres et le label
useful_features_label = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                         'dst_host_rerror_rate', 'dst_bytes', 'hot',
                         'num_failed_logins', 'dst_host_srv_count', 'label']

# Les 8 features utiles sans la feature label
useful_features = ['service', 'src_bytes', 'dst_host_diff_srv_rate',
                   'dst_host_rerror_rate', 'dst_bytes', 'hot',
                   'num_failed_logins', 'dst_host_srv_count']

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
    'DataPath', 'train_data_path'), names=feature_name)
test_dataframe = pd.read_csv(config_parser.get(
    'DataPath', 'test_data_path'), names=feature_name)


def process_dataframe(dataframe, name):
    # Réduit le dataframe en ne conservant que les features "utiles"
    dataframe = dataframe[useful_features_label]

    # Récupère le nombre de chaque type de données pour afficher des stats
    total_data_length = len(dataframe)
    normal_data_length = len(dataframe[dataframe['label'] == 'normal.'])
    anormal_data_length = len(dataframe[dataframe['label'] != 'normal.'])

    def print_data(attack_name, attack_lengh):
        print(' '*6 + attack_name + ': {:,} soit {}%'
              .format(attack_lengh,
                      round(attack_lengh * 100 / total_data_length, 3))
              .replace(',', ' '))

    print('\nDataset {}:'.format(name))
    print('Total Entrees: {:,}'.format(total_data_length).replace(',', ' '))
    print_data("Entrees normales", normal_data_length)
    print_data("Entrees anormales", anormal_data_length)

    # Assigne numero différent selon la nature de la connexion
    dataframe.loc[dataframe['label'] == 'normal.', 'label'] = 0
    for i in range(len(probe)):
        dataframe.loc[dataframe['label'] == probe[i], 'label'] = 1
    for i in range(len(dos)):
        dataframe.loc[dataframe['label'] == dos[i], 'label'] = 2
    for i in range(len(u2r)):
        dataframe.loc[dataframe['label'] == u2r[i], 'label'] = 3
    for i in range(len(r2l)):
        dataframe.loc[dataframe['label'] == r2l[i], 'label'] = 4

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

    # # Ajoute chaque nouvelle valeur de service dans la liste
    # for i in range(len(dataframe['service'])):
    #     if dataframe['service'][i] not in service_values:
    #         service_values.append(dataframe['service'][i])

    # Convertit la valeur du service en integer en focntion de sa position
    for i in range(len(service_values)):
        dataframe.loc[dataframe['service'] == service_values[i], 'service'] = i

    # Création des tableaux d'entrées X et de sortie y
    x = np.array(dataframe[useful_features])
    y = np.array(dataframe['label'])

    # Encodage one-hot de y (=> valeur 0 ou 1)
    # Tensorflow n'accepte que des labels encodés one-hot
    y_one_hot = np.zeros([y.shape[0], 5])
    for i in range(y.shape[0]):
        if y[i] == 0:
            y_one_hot[i, 0] = 1
        elif y[i] == 2:
            y_one_hot[i, 2] = 1
        elif y[i] == 3:
            y_one_hot[i, 3] = 1
        elif y[i] == 4:
            y_one_hot[i, 4] = 1
        elif y[i] == 5:
            y_one_hot[i, 5] = 1

    # Standardise les donnees en les centrant et les mettant à l'echelle
    # à partir de la moyenne et de l'écart-type
    standardScaler = StandardScaler()
    standardScaler.fit(x)
    x = standardScaler.transform(x)
    x = x.reshape([-1, x.shape[1], 1])

    X, Y = x[:total_data_length, :], y_one_hot[:total_data_length, :]

    return X, Y


x_train, y_train = process_dataframe(train_dataframe, 'entrainement')
x_test, y_test = process_dataframe(test_dataframe, 'test')

np.save(config_parser.get('DataPath', 'x_train_path'), x_train)
np.save(config_parser.get('DataPath', 'y_train_path'), y_train)
np.save(config_parser.get('DataPath', 'x_test_path'), x_test)
np.save(config_parser.get('DataPath', 'y_test_path'), y_test)

print("\nEnregistrement des datasets fini.")
