from context_printer import Color
from context_printer import ContextPrinter as Ctp

import pandas as pd
import torch
import torch.utils.data

import numpy as np

all_devices = ['Danmini_Doorbell',
               'Ecobee_Thermostat',
               'Ennio_Doorbell',
               'Philips_B120N10_Baby_Monitor',
               'Provision_PT_737E_Security_Camera',
               'Provision_PT_838_Security_Camera',
               'Samsung_SNH_1011_N_Webcam',
               'SimpleHome_XCS7_1002_WHT_Security_Camera',
               'SimpleHome_XCS7_1003_WHT_Security_Camera']

# List of the devices that can be infected by the mirai malware
mirai_devices = all_devices[0:2] + all_devices[3:6] + all_devices[7:9]

mirai_attacks = ['ack', 'scan', 'syn', 'udp', 'udpplain']
gafgyt_attacks = ['combo', 'junk', 'scan', 'tcp', 'udp']

data_path = 'data/N-BaIoT/'

benign_paths = {device: data_path + device + '/benign_traffic.csv' for device in all_devices}

mirai_paths = [{device: data_path + device + '/mirai_attacks/' + attack + '.csv' for device in mirai_devices}
               for attack in mirai_attacks]

gafgyt_paths = [{device: data_path + device + '/gafgyt_attacks/' + attack + '.csv' for device in all_devices}
                for attack in gafgyt_attacks]


def device_names(device_ids):
    return ', '.join([all_devices[device_id] for device_id in device_ids])


def get_sub_div(data, normalization):
    if normalization == '0-mean 1-var':
        sub = data.mean(dim=0)
        div = data.std(dim=0)
    elif normalization == 'min-max':
        sub = data.min(dim=0)[0]
        div = data.max(dim=0)[0] - sub
    elif normalization == 'none':
        sub = torch.zeros(data.shape[1])
        div = torch.ones(data.shape[1])
    else:
        raise NotImplementedError

    return sub, div


def get_device_dataframes(device_id: int):
    device = all_devices[device_id]
    dataframes = {'benign': pd.read_csv(benign_paths[device])}
    if device in mirai_devices:
        dataframes.update({'mirai_' + attack: pd.read_csv(attack_paths[device]) for attack, attack_paths in zip(mirai_attacks, mirai_paths)})

    dataframes.update({'gafgyt_' + attack: pd.read_csv(attack_paths[device]) for attack, attack_paths in zip(gafgyt_attacks, gafgyt_paths)})
    return dataframes


# Note that if the number of rows in a dataframe is not a multiple of the number of splits, some rows will be left out of all splits
def get_splits(device_dataframes: dict, splits_benign: list, splits_attack: list) -> dict:
    # Compute the indexes of the splits.
    # For example for benign data, with splits_benign = [0.3, 0.7] and device_dataframes['benign'] of length 100,
    # we would have indexes['benign'] = [0, 30, 100]
    indexes = {'benign': [0] + list(np.cumsum([int(split_proportion * len(device_dataframes['benign'])) for split_proportion in splits_benign]))}
    indexes.update({key: [0] + list(np.cumsum([int(split_proportion * len(device_dataframes[key])) for split_proportion in splits_attack]))
                    for key in device_dataframes.keys() if key != 'benign'})

    Ctp.print(str(indexes))

    # For each key (benign, mirai_ack, ...) and for each split we take the corresponding rows of the corresponding dataframe
    data_splits = {key: [torch.tensor(device_dataframes[key].values[indexes[key][split_id]:indexes[key][split_id+1]]).float()
                         for split_id in range(len(indexes[key]) - 1)]
                   for key in device_dataframes.keys()}
    return data_splits


def get_supervised_datasets(devices_dataframes: list):
    train_data, test_data, train_targets, test_targets = [], [], [], []
    for device_dataframes in devices_dataframes:
        data_splits = get_splits(device_dataframes, splits_benign=[0.5, 0.5], splits_attack=[0.5, 0.5])
        for key, splits in data_splits.items():  # This will iterate over the benign splits, gafgyt splits and mirai splits (if applicable)
            train_data.append(splits[0])
            test_data.append(splits[1])
            train_targets.append(torch.full((splits[0].shape[0], 1), (0. if key == 'benign' else 1.)))
            test_targets.append(torch.full((splits[1].shape[0], 1), (0. if key == 'benign' else 1.)))

    dataset_train = torch.utils.data.TensorDataset(torch.cat(train_data, dim=0), torch.cat(train_targets, dim=0))
    dataset_test = torch.utils.data.TensorDataset(torch.cat(test_data, dim=0), torch.cat(test_targets, dim=0))

    return dataset_train, dataset_test


def get_unsupervised_datasets(devices_dataframes: list):
    train_data, opt_data = [], []
    test_data = {**{'benign': []},
                 **{'mirai_' + attack: [] for attack in mirai_attacks},
                 **{'gafgyt_' + attack: [] for attack in gafgyt_attacks}}

    for device_dataframes in devices_dataframes:
        data_splits = get_splits(device_dataframes, splits_benign=[0.33, 0.33, 0.33], splits_attack=[1.0])
        print(len(data_splits))
        train_data.append(data_splits['benign'][0])
        opt_data.append(data_splits['benign'][1])
        test_data['benign'].append(data_splits['benign'][2])
        for key, splits in data_splits.items():
            if key != 'benign':
                test_data[key].append(splits[0])

    dataset_train = torch.utils.data.TensorDataset(torch.cat(train_data, dim=0))
    dataset_opt = torch.utils.data.TensorDataset(torch.cat(opt_data, dim=0))
    datasets_test = {key: torch.utils.data.TensorDataset(torch.cat(test_data[key], dim=0)) for key in test_data.keys() if len(test_data[key]) > 0}
    # datasets_test is a dict of attack name (or "benign") to dataset

    return dataset_train, dataset_opt, datasets_test


def get_device_ids(clients_devices, test_devices):
    device_ids = []
    for client_device_ids in clients_devices:
        for device_id in client_device_ids:
            if device_id not in device_ids:
                device_ids.append(device_id)

    for device_id in test_devices:
        if device_id not in device_ids:
            device_ids.append(device_id)
    return device_ids


def get_dataframes(color=Color.NONE):
    Ctp.enter_section('Reading data', color)

    device_id_to_dataframes = {}
    for device_id in range(len(all_devices)):
        Ctp.print('[{}/{}] Data from '.format(device_id + 1, len(all_devices)) + all_devices[device_id])
        device_id_to_dataframes.update({device_id: get_device_dataframes(device_id)})
    Ctp.exit_section()

    return device_id_to_dataframes


def get_client_supervised_dataloaders(args, client_device_ids, device_id_to_dataframes):
    devices_dataframes = [device_id_to_dataframes[device_id] for device_id in client_device_ids]
    dataset_train, dataset_test = get_supervised_datasets(devices_dataframes)
    client_dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_bs, shuffle=True)
    client_dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_bs)
    return client_dl_train, client_dl_test


# args.clients_devices should be a list of lists of devices and args.test_devices should be a list of devices
def get_supervised_dataloaders(args, device_id_to_dataframes: dict):
    # Step 1: create the datasets and the dataloaders of the clients: 1 train and 1 test per client
    clients_dl_train, clients_dl_test = [], []
    for client_device_ids in args.clients_devices:
        client_dl_train, client_dl_test = get_client_supervised_dataloaders(args, client_device_ids, device_id_to_dataframes)
        clients_dl_train.append(client_dl_train)
        clients_dl_test.append(client_dl_test)

    # Step 2: create the dataset and the dataloader of the new devices (test only)
    _, new_dl_test = get_client_supervised_dataloaders(args, args.test_devices, device_id_to_dataframes)

    Ctp.exit_section()
    return clients_dl_train, clients_dl_test, new_dl_test


# args.clients_devices should be a list of lists of devices and args.test_devices should be a list of devices
def get_supervised_dataloaders_v2(args, device_id_to_dataframes: dict, proportion_test: float, proportion_unused: float = 0., test=False, n_folds=1,
                                  fold=1, ):
    if test:
        pass
        # Clients train on (1 - proportion_test - proportion_unused) of their data
        # Clients test on proportion_test of their data
        # Global model is also tested on proportion_test of the test devices' data

    else:
        pass
        # Clients train on (1 - proportion_test - proportion_unused) * (k-1) / k of their data
        # Clients test on (1 - proportion_test - proportion_unused) * 1 / k of their data
        # Global model is also tested on (1 - proportion_test - proportion_unused) of the test devices' data

    # Step 1: create the datasets and the dataloaders of the clients: 1 train and 1 test per client
    clients_dl_train, clients_dl_test = [], []
    for client_device_ids in args.clients_devices:
        client_dl_train, client_dl_test = get_client_supervised_dataloaders(args, client_device_ids, device_id_to_dataframes)
        clients_dl_train.append(client_dl_train)
        clients_dl_test.append(client_dl_test)

    # Step 2: create the dataset and the dataloader of the new devices (test only)
    _, new_dl_test = get_client_supervised_dataloaders(args, args.test_devices, device_id_to_dataframes)

    Ctp.exit_section()
    return clients_dl_train, clients_dl_test, new_dl_test


def get_client_unsupervised_dataloaders(args, client_device_ids, device_id_to_dataframes):
    devices_dataframes = [device_id_to_dataframes[device_id] for device_id in client_device_ids]
    dataset_train, dataset_opt, datasets_test = get_unsupervised_datasets(devices_dataframes)
    client_dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_bs, shuffle=True)
    client_dl_opt = torch.utils.data.DataLoader(dataset_opt, batch_size=args.test_bs)
    client_dls_test = {key: torch.utils.data.DataLoader(datasets_test[key], batch_size=args.test_bs) for key in datasets_test.keys()}
    return client_dl_train, client_dl_opt, client_dls_test


# args.clients_devices should be a list of lists of devices and args.test_devices should be a list of devices
def get_unsupervised_dataloaders(args, device_id_to_dataframes: dict):
    # Step 1: create the datasets and the dataloaders of the clients: 1 train, 1 opt and 1 dict of test dataloaders per client
    clients_dl_train, clients_dl_opt, clients_dls_test = [], [], []
    for client_device_ids in args.clients_devices:
        client_dl_train, client_dl_opt, client_dls_test = get_client_unsupervised_dataloaders(args, client_device_ids, device_id_to_dataframes)
        clients_dl_train.append(client_dl_train)
        clients_dl_opt.append(client_dl_opt)
        clients_dls_test.append(client_dls_test)

    # Step 2: create the dataset and the dataloader of the new devices (test only)
    _, _, new_dls_test = get_client_unsupervised_dataloaders(args, args.test_devices, device_id_to_dataframes)

    return clients_dl_train, clients_dl_opt, clients_dls_test, new_dls_test

# For supervised (same procedure applies to benign and attack data)
# Inputs:
# number of folds k
# percentage of data left to the test set p
# args
# color
#
# Procedure
# Load the data from memory
# Separate the data between the test set (based on p) and the k folds
# this part should be made with a simple call to get_splits, but this function needs to be changed to allow that not all subsets have the same size
# From that, create the k train/val datasets and the test set, return them as a list of datasets called datasets_train and a single dataset_test
# From that we can create the k dataloaders with all variations of k-1 train sets and 1 val set. The output should be for each client:
# a list of tuples (client_dl_train, client_dl_val) of size k, and a single client_dl_test
# From that we can compute that for all clients and return them similarly as a list of size number of clients of
#
# Create k tion datasets, 1 test set
# Return value
