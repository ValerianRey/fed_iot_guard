from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset


def get_dataset(data: List[Dict[str, np.array]]) -> Dataset:
    data_list, target_list = [], []
    for device_data in data:
        for key, arr in device_data.items():  # This will iterate over the benign splits, gafgyt splits and mirai splits (if applicable)
            data_list.append(torch.tensor(arr).float())
            target_list.append(torch.full((arr.shape[0], 1), (0. if key == 'benign' else 1.)))
    dataset = TensorDataset(torch.cat(data_list, dim=0), torch.cat(target_list, dim=0))
    return dataset


def get_client_dls(client_device_ids: List[int], train_data: List[Dict[str, np.array]], test_data: List[Dict[str, np.array]],
                   train_bs: int, test_bs: int) -> Tuple[DataLoader, DataLoader]:

    client_train_data = [train_data[device_id] for device_id in client_device_ids]
    client_test_data = [test_data[device_id] for device_id in client_device_ids]
    dataset_train = get_dataset(client_train_data)
    dataset_test = get_dataset(client_test_data)
    client_dl_train = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    client_dl_test = DataLoader(dataset_test, batch_size=test_bs)
    return client_dl_train, client_dl_test


def get_all_supervised_dls(train_data: List[Dict[str, np.array]], test_data: List[Dict[str, np.array]],
                           clients_devices: List[List[int]], test_devices: List[int], train_bs: int, test_bs: int)\
        -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:

    # Step 1: create the datasets and the dataloaders of the clients: 1 train and 1 test per client
    clients_dl_train, clients_dl_test = [], []
    for client_device_ids in clients_devices:
        client_dl_train, client_dl_test = get_client_dls(client_device_ids, train_data, test_data, train_bs, test_bs)
        clients_dl_train.append(client_dl_train)
        clients_dl_test.append(client_dl_test)

    # Step 2: create the dataset and the dataloaders of the new devices (test only)
    _, new_dl_test = get_client_dls(test_devices, train_data, test_data, train_bs, test_bs)

    return clients_dl_train, clients_dl_test, new_dl_test
