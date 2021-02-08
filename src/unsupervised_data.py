from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data import mirai_attacks, gafgyt_attacks


def get_train_dataset(train_data: List[Dict[str, np.array]]) -> Dataset:
    data_list = [torch.tensor(device_data['benign']).float() for device_data in train_data]
    dataset = TensorDataset(torch.cat(data_list, dim=0))
    return dataset


def get_test_datasets(test_data: List[Dict[str, np.array]]) -> Dict[str, Dataset]:
    data_dict = {**{'benign': []},
                 **{'mirai_' + attack: [] for attack in mirai_attacks},
                 **{'gafgyt_' + attack: [] for attack in gafgyt_attacks}}

    for device_data in test_data:
        for key, arr in device_data.items():
            data_dict[key].append(torch.tensor(arr).float())

    datasets_test = {key: TensorDataset(torch.cat(data_dict[key], dim=0)) for key in data_dict.keys() if len(data_dict[key]) > 0}
    return datasets_test


def get_client_dls(client_device_ids: List[int], train_data: List[Dict[str, np.array]], opt_data: List[Dict[str, np.array]],
                   test_data: List[Dict[str, np.array]], train_bs: int, test_bs: int) \
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


def get_all_unsupervised_dls(train_data: List[Dict[str, np.array]], opt_data: List[Dict[str, np.array]],
                             test_data: List[Dict[str, np.array]], clients_devices: List[List[int]], test_devices: List[int],
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
