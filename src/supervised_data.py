from typing import List, Tuple

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import multiclass_labels, ClientData, FederationData, split_client_data


def get_target_tensor(key: str, arr: np.ndarray, multiclass: bool = False) -> torch.Tensor:
    if multiclass:
        return torch.full((arr.shape[0], 1), multiclass_labels[key])
    else:
        return torch.full((arr.shape[0], 1), (0. if key == 'benign' else 1.))


def get_dataset(data: ClientData, cuda: bool = False, multiclass: bool = False) -> Dataset:
    data_list, target_list = [], []
    for device_data in data:
        for key, arr in device_data.items():  # This will iterate over the benign splits, gafgyt splits and mirai splits (if applicable)
            data_tensor = torch.tensor(arr).float()
            target_tensor = get_target_tensor(key, arr, multiclass=multiclass)
            if cuda:
                data_tensor = data_tensor.cuda()
                target_tensor = target_tensor.cuda()
            data_list.append(data_tensor)
            target_list.append(target_tensor)
    dataset = TensorDataset(torch.cat(data_list, dim=0), torch.cat(target_list, dim=0))
    return dataset


def get_train_dl(client_train_data: ClientData, train_bs: int, cuda: bool = False, multiclass: bool = False) -> DataLoader:
    dataset_train = get_dataset(client_train_data, cuda=cuda, multiclass=multiclass)
    train_dl = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    return train_dl


def get_test_dl(client_test_data: ClientData, test_bs: int, cuda: bool = False, multiclass: bool = False) -> DataLoader:
    dataset_test = get_dataset(client_test_data, cuda=cuda, multiclass=multiclass)
    test_dl = DataLoader(dataset_test, batch_size=test_bs)
    return test_dl


def get_train_test_dls(train_data: FederationData, test_data: FederationData, train_bs: int, test_bs: int,
                       cuda: bool = False, multiclass: bool = False) -> Tuple[List[DataLoader], List[DataLoader]]:

    train_dls = [get_train_dl(client_train_data, train_bs, cuda=cuda, multiclass=multiclass) for client_train_data in train_data]
    test_dls = [get_test_dl(client_test_data, test_bs, cuda=cuda, multiclass=multiclass) for client_test_data in test_data]
    return train_dls, test_dls


def get_client_supervised_initial_splitting(client_data: ClientData, p_test: float, p_unused: float) -> Tuple[ClientData, ClientData]:
    client_train_val, client_test = split_client_data(client_data, p_test=p_test, p_unused=p_unused)
    return client_train_val, client_test
