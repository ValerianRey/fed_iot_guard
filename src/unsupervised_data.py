from typing import Dict, Tuple, List, Optional

import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import mirai_attacks, gafgyt_attacks, split_client_data, ClientData, FederationData, compute_alphas_resampling, resample_array


def get_benign_dataset(train_data: ClientData, cuda: bool = False) -> Dataset:
    data_list = [torch.tensor(device_data['benign']).float() for device_data in train_data]
    if cuda:
        data_list = [tensor.cuda() for tensor in data_list]
    dataset = TensorDataset(torch.cat(data_list, dim=0))
    return dataset


def get_test_datasets(test_data: ClientData, sampling: Optional[str] = None, p_benign: Optional[float] = None,
                      cuda: bool = False) -> Dict[str, Dataset]:
    data_dict = {**{'benign': []},
                 **{'mirai_' + attack: [] for attack in mirai_attacks},
                 **{'gafgyt_' + attack: [] for attack in gafgyt_attacks}}

    for device_data in test_data:
        benign_samples = sum([len(arr) for key, arr in device_data.items() if key == 'benign'])
        attack_samples = sum([len(arr) for key, arr in device_data.items() if key != 'benign'])
        alpha_benign, alpha_attack = compute_alphas_resampling(benign_samples, attack_samples, sampling, p_benign, verbose=True)

        for key, arr in device_data.items():
            if key == 'benign':
                arr = resample_array(arr, alpha_benign)
            else:
                arr = resample_array(arr, alpha_attack)

            data_tensor = torch.tensor(arr).float()
            if cuda:
                data_tensor = data_tensor.cuda()
            data_dict[key].append(data_tensor)

    datasets_test = {key: TensorDataset(torch.cat(data_dict[key], dim=0)) for key in data_dict.keys() if len(data_dict[key]) > 0}
    return datasets_test


def get_train_dl(client_train_data: ClientData, train_bs: int, cuda: bool = False) -> DataLoader:
    dataset_train = get_benign_dataset(client_train_data, cuda=cuda)
    train_dl = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    return train_dl


def get_val_dl(client_val_data: ClientData, test_bs: int, cuda: bool = False) -> DataLoader:
    dataset_val = get_benign_dataset(client_val_data, cuda=cuda)
    val_dl = DataLoader(dataset_val, batch_size=test_bs)
    return val_dl


def get_test_dls_dict(client_test_data: ClientData, test_bs: int, sampling: Optional[str] = None, p_benign: Optional[float] = None,
                      cuda: bool = False) -> Dict[str, DataLoader]:
    datasets = get_test_datasets(client_test_data, sampling=sampling, p_benign=p_benign, cuda=cuda)
    test_dls = {key: DataLoader(dataset, batch_size=test_bs) for key, dataset in datasets.items()}
    return test_dls


def restrict_new_device_benign_data(new_device_data: ClientData, p_test: float, sampling: Optional[str] = None) -> None:
    if sampling is None:  # Otherwise we already handle the dataset balance somewhere else
        for device_data in new_device_data:
            for key, arr in device_data.items():
                if key == 'benign':
                    begin_index = int(len(arr) * (1 - p_test))
                    device_data[key] = arr[begin_index:]


def get_train_val_test_dls(train_data: FederationData, val_data: FederationData, local_test_data: FederationData, train_bs: int, test_bs: int,
                           sampling: Optional[str] = None, p_benign: Optional[float] = None,
                           cuda: bool = False) -> Tuple[List[DataLoader], List[DataLoader], List[Dict[str, DataLoader]]]:
    clients_dl_train = [get_train_dl(client_train_data, train_bs, cuda=cuda) for client_train_data in train_data]
    clients_dl_val = [get_val_dl(client_val_data, test_bs, cuda=cuda) for client_val_data in val_data]
    clients_dls_test = [get_test_dls_dict(client_test_data, test_bs, sampling=sampling, p_benign=p_benign, cuda=cuda)
                        for client_test_data in local_test_data]

    return clients_dl_train, clients_dl_val, clients_dls_test


def get_client_unsupervised_initial_splitting(client_data: ClientData, p_test: float, p_unused: float) -> Tuple[ClientData, ClientData]:
    # Separate the data of the clients between benign and attack
    client_benign_data = [{'benign': device_data['benign']} for device_data in client_data]
    client_attack_data = [{key: device_data[key] for key in device_data.keys() if key != 'benign'} for device_data in client_data]
    client_train_val, client_benign_test = split_client_data(client_benign_data, p_test=p_test, p_unused=p_unused)
    client_test = [{**device_benign_test, **device_attack_data}
                   for device_benign_test, device_attack_data in zip(client_benign_test, client_attack_data)]

    return client_train_val, client_test
