import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,
                                   LabelEncoder, MinMaxScaler)
pd.options.mode.chained_assignment = None  # default='warn' | Disable warnings

# ***** UNSW STRING FEATURES VALUES *****
proto_values = ['tcp', 'udp', 'arp', 'ospf', 'icmp', 'igmp', 'rtp', 'ddp',
                'ipv6-frag', 'cftp', 'wsn', 'pvp', 'wb-expak', 'mtp',
                'pri-enc', 'sat-mon', 'cphb', 'sun-nd', 'iso-ip', 'xtp', 'il',
                'unas', 'mfe-nsp', '3pc', 'ipv6-route', 'idrp', 'bna', 'swipe',
                'kryptolan', 'cpnx', 'rsvp', 'wb-mon', 'vmtp', 'ib', 'dgp',
                'eigrp', 'ax.25', 'gmtp', 'pnni', 'sep', 'pgm', 'idpr-cmtp',
                'zero', 'rvd', 'mobile', 'narp', 'fc', 'pipe', 'ipcomp',
                'ipv6-no', 'sat-expak', 'ipv6-opts', 'snp', 'ipcv',
                'br-sat-mon', 'ttp', 'tcf', 'nsfnet-igp', 'sprite-rpc',
                'aes-sp3-d', 'sccopmce', 'sctp', 'qnx', 'scps', 'etherip',
                'aris', 'pim', 'compaq-peer', 'vrrp', 'iatp', 'stp',
                'l2tp', 'srp', 'sm', 'isis', 'smp', 'fire', 'ptp', 'crtp',
                'sps', 'merit-inp', 'idpr', 'skip', 'any', 'larp', 'ipip',
                'micp', 'encap', 'ifmp', 'tp++', 'a/n', 'ipv6', 'i-nlsp',
                'ipx-n-ip', 'sdrp', 'tlsp', 'gre', 'mhrp', 'ddx', 'ippc',
                'visa', 'secure-vmtp', 'uti', 'vines', 'crudp', 'iplt',
                'ggp', 'ip', 'ipnip', 'st2', 'argus', 'bbn-rcc', 'egp',
                'emcon', 'igp', 'nvp', 'pup', 'xnet', 'chaos', 'mux', 'dcn',
                'hmp', 'prm', 'trunk-1', 'xns-idp', 'leaf-1', 'leaf-2', 'rdp',
                'irtp', 'iso-tp4', 'netblt', 'trunk-2', 'cbt']

state_values = ['FIN', 'INT', 'CON', 'ECO', 'REQ', 'RST', 'PAR', 'URN', 'no',
                'ACC', 'CLO']

service_values = ['-', 'ftp', 'smtp', 'snmp', 'http', 'ftp-data',
                  'dns', 'ssh', 'radius', 'pop3', 'dhcp', 'ssl', 'irc']

attack_cat_values = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode',
                     'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']


def unsw_encoding(params):
    # ***** DATA PATHS *****
    data_path = "./data/"
    train_data_path = data_path+"UNSW_NB15_training-set.csv"
    test_data_path = data_path+"UNSW_NB15_testing-set.csv"

    # Load csv data into dataframes without 'id' and 'Label'
    train_df = pd.read_csv(train_data_path).drop(['id', 'label'], axis=1)
    test_df = pd.read_csv(test_data_path).drop(['id', 'label'], axis=1)

    def process_dataframe(df):
        # Replace attack string with an int
        for i in range(len(attack_cat_values)):
            df['attack_cat'] = df['attack_cat'].replace(
                [attack_cat_values[i]], i)

        # Assign x (inputs) and y (outputs) of the network
        y = df['attack_cat']
        x = df.drop(columns='attack_cat')

        # ***** MULTIPLE ENCODER CHOICE *****
        # Encode categorical features as an integer array
        if params['encoder'] == 'ordinalencoder':
            x = OrdinalEncoder().fit_transform(x)
        # Encode labels with value between 0 and n_classes-1.
        elif params['encoder'] == 'labelencoder':
            x = x.apply(LabelEncoder().fit_transform)
        else:
            # Replace String features with ints
            for i in range(len(proto_values)):
                x['proto'] = x['proto'].replace(proto_values[i], i)

            for i in range(len(state_values)):
                x['state'] = x['state'].replace(state_values[i], i)

            for i in range(len(service_values)):
                x['service'] = x['service'].replace(service_values[i], i)
            # Standardize by removing the mean and scaling to unit variance
            if params['encoder'] == "standardscaler":
                x = StandardScaler().fit_transform(x)
            # Transforms features by scaling each feature to range [0, 1]
            elif params['encoder'] == "minmaxscaler01":
                x = MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
            # Transforms features by scaling each feature to range [-1, 1]
            elif params['encoder'] == "minmaxscaler11":
                x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)

        return x, y

    x_train, Y_train = process_dataframe(train_df)
    x_test, Y_test = process_dataframe(test_df)

    # Apply one-hot encoding to outputs
    y_train = to_categorical(Y_train)
    y_test = to_categorical(Y_test)

    return x_train, x_test, y_train, y_test
