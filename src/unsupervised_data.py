from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data import mirai_attacks, gafgyt_attacks, get_client_data_combined, split_data


def get_train_dataset(train_data: List[Dict[str, np.ndarray]]) -> Dataset:
    data_list = [torch.tensor(device_data['benign']).float() for device_data in train_data]
    dataset = TensorDataset(torch.cat(data_list, dim=0))
    return dataset


def get_test_datasets(test_data: List[Dict[str, np.ndarray]]) -> Dict[str, Dataset]:
    data_dict = {**{'benign': []},
                 **{'mirai_' + attack: [] for attack in mirai_attacks},
                 **{'gafgyt_' + attack: [] for attack in gafgyt_attacks}}

    for device_data in test_data:
        for key, arr in device_data.items():
            data_dict[key].append(torch.tensor(arr).float())

    datasets_test = {key: TensorDataset(torch.cat(data_dict[key], dim=0)) for key in data_dict.keys() if len(data_dict[key]) > 0}
    return datasets_test


def get_client_dls(client_device_ids: List[int], train_data: List[Dict[str, np.ndarray]], opt_data: List[Dict[str, np.ndarray]],
                   test_data: List[Dict[str, np.ndarray]], train_bs: int, test_bs: int) \
        -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
    client_train_data = [train_data[device_id] for device_id in client_device_ids]
    client_opt_data = [opt_data[device_id] for device_id in client_device_ids]
    client_test_data = [test_data[device_id] for device_id in client_device_ids]
    dataset_train = get_train_dataset(client_train_data)
    dataset_opt = get_train_dataset(client_opt_data)
    datasets_test = get_test_datasets(client_test_data)
    client_dl_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    client_dl_opt = DataLoader(dataset_opt, batch_size=test_bs)
    client_dls_test = {key: DataLoader(datasets_test[key], batch_size=test_bs) for key in datasets_test.keys()}
    return client_dl_train, client_dl_opt, client_dls_test


def get_all_unsupervised_dls(train_data: List[Dict[str, np.ndarray]], opt_data: List[Dict[str, np.ndarray]],
                             test_data: List[Dict[str, np.ndarray]], clients_devices: List[List[int]], test_devices: List[int],
                             train_bs: int, test_bs: int) \
        -> Tuple[List[DataLoader], List[DataLoader], List[Dict[str, DataLoader]], Dict[str, DataLoader]]:
    # Step 1: create the datasets and the dataloaders of the clients: 1 train, 1 opt and 1 dict of test dataloaders per client
    clients_dl_train, clients_dl_opt, clients_dls_test = [], [], []
    for client_device_ids in clients_devices:
        client_dl_train, client_dl_opt, client_dls_test = get_client_dls(client_device_ids, train_data,
                                                                         opt_data, test_data, train_bs, test_bs)
        clients_dl_train.append(client_dl_train)
        clients_dl_opt.append(client_dl_opt)
        clients_dls_test.append(client_dls_test)

    # Step 2: create the dataset and the dataloader of the new devices (test only)
    _, _, new_dls_test = get_client_dls(test_devices, train_data, opt_data, test_data, train_bs, test_bs)

    return clients_dl_train, clients_dl_opt, clients_dls_test, new_dls_test


def get_client_train_opt_dls(client_device_ids: List[int], train_data: List[Dict[str, np.ndarray]], opt_data: List[Dict[str, np.ndarray]],
                             train_bs: int, opt_bs: int) -> Tuple[DataLoader, DataLoader]:
    client_train_data = [train_data[device_id] for device_id in client_device_ids]
    client_opt_data = [opt_data[device_id] for device_id in client_device_ids]
    dataset_train = get_train_dataset(client_train_data)
    dataset_opt = get_train_dataset(client_opt_data)
    client_dl_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    client_dl_opt = DataLoader(dataset_opt, batch_size=opt_bs)
    return client_dl_train, client_dl_opt


def get_all_train_opt_dls(train_data: List[Dict[str, np.ndarray]], opt_data: List[Dict[str, np.ndarray]], clients_devices: List[List[int]],
                          train_bs: int, opt_bs: int) -> Tuple[List[DataLoader], List[DataLoader]]:
    # Step 1: create the datasets and the dataloaders of the clients: 1 train, 1 opt and 1 dict of test dataloaders per client
    clients_dl_train, clients_dl_opt = [], []
    for client_device_ids in clients_devices:
        client_dl_train, client_dl_opt = get_client_train_opt_dls(client_device_ids, train_data,
                                                                  opt_data, train_bs, opt_bs)
        clients_dl_train.append(client_dl_train)
        clients_dl_opt.append(client_dl_opt)

    return clients_dl_train, clients_dl_opt


def get_initial_splitting(clients_devices_data: List[List[Dict[str, np.ndarray]]], test_devices_data: List[Dict[str, np.ndarray]],
                          p_test: float, p_unused: float) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:

    # Flatten each client's data: clients_data is now a list of dicts and new_devices_test is now a simple dict
    clients_data = [get_client_data_combined(client_devices_data) for client_devices_data in clients_devices_data]
    new_devices_test = get_client_data_combined(test_devices_data)

    # Separate the data of the clients between benign and attack
    clients_benign_data = [{'benign': client_data['benign']} for client_data in clients_data]
    clients_attack_data = [{key: client_data[key] for key in client_data.keys() if key != 'benign'} for client_data in clients_data]

    # Split the benign data between train/validation and test, leaving an unused set in between
    clients_train_val, clients_benign_test = split_data(clients_benign_data, p_test=p_test, p_unused=p_unused)

    # The local tests are made with the benign test sets and all attack sets
    clients_test = [{**client_benign_test, **client_attack_data}
                    for client_benign_test, client_attack_data in zip(clients_benign_test, clients_attack_data)]

    return clients_train_val, clients_test, new_devices_test
