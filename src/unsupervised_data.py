from types import SimpleNamespace
from typing import Dict, Tuple, List, Optional

import torch
import torch.utils
import torch.utils.data
# noinspection PyProtectedMember
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import mirai_attacks, gafgyt_attacks, split_client_data, ClientData, FederationData, resample_array, split_clients_data, \
    get_benign_attack_samples_per_device


def get_benign_dataset(data: ClientData, benign_samples_per_device: Optional[int] = None, cuda: bool = False) -> Dataset:
    data_list = []
    for device_data in data:
        for key, arr in device_data.items():  # This will iterate over the benign splits, gafgyt splits and mirai splits (if applicable)
            if key == 'benign':
                if benign_samples_per_device is not None:
                    arr = resample_array(arr, benign_samples_per_device)

            data_tensor = torch.tensor(arr).float()
            if cuda:
                data_tensor = data_tensor.cuda()
            data_list.append(data_tensor)

    dataset = TensorDataset(torch.cat(data_list, dim=0))
    return dataset


def get_test_datasets(test_data: ClientData, benign_samples_per_device: Optional[int] = None, attack_samples_per_device: Optional[int] = None,
                      cuda: bool = False) -> Dict[str, Dataset]:
    data_dict = {**{'benign': []},
                 **{'mirai_' + attack: [] for attack in mirai_attacks},
                 **{'gafgyt_' + attack: [] for attack in gafgyt_attacks}}

    resample = benign_samples_per_device is not None and attack_samples_per_device is not None

    for device_data in test_data:
        if resample:
            number_of_attacks = len(device_data.keys()) - 1
            n_samples_attack = attack_samples_per_device // 10
            if number_of_attacks == 5:
                n_samples_attack *= 2
        else:
            n_samples_attack = None
        # We evenly divide the attack samples among the existing attacks on that device
        # With the above trick we always end up with exactly the same total number of attack samples,
        # whether the device has 5 attacks or 10. It does not work if the device has any other number of attacks,
        # but with N-BaIoT this is never the case

        for key, arr in device_data.items():
            if resample:
                if key == 'benign':
                    arr = resample_array(arr, benign_samples_per_device)
                else:
                    arr = resample_array(arr, n_samples_attack)

            data_tensor = torch.tensor(arr).float()
            if cuda:
                data_tensor = data_tensor.cuda()
            data_dict[key].append(data_tensor)

    datasets_test = {key: TensorDataset(torch.cat(data_dict[key], dim=0)) for key in data_dict.keys() if len(data_dict[key]) > 0}
    return datasets_test


def get_train_dl(client_train_data: ClientData, train_bs: int, benign_samples_per_device: Optional[int] = None, cuda: bool = False) -> DataLoader:
    dataset_train = get_benign_dataset(client_train_data, benign_samples_per_device=benign_samples_per_device, cuda=cuda)
    train_dl = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    return train_dl


def get_val_dl(client_val_data: ClientData, test_bs: int, benign_samples_per_device: Optional[int] = None, cuda: bool = False) -> DataLoader:
    dataset_val = get_benign_dataset(client_val_data, benign_samples_per_device=benign_samples_per_device, cuda=cuda)
    val_dl = DataLoader(dataset_val, batch_size=test_bs)
    return val_dl


def get_test_dls_dict(client_test_data: ClientData, test_bs: int, benign_samples_per_device: Optional[int] = None,
                      attack_samples_per_device: Optional[int] = None, cuda: bool = False) -> Dict[str, DataLoader]:
    datasets = get_test_datasets(client_test_data, benign_samples_per_device=benign_samples_per_device,
                                 attack_samples_per_device=attack_samples_per_device, cuda=cuda)
    test_dls = {key: DataLoader(dataset, batch_size=test_bs) for key, dataset in datasets.items()}
    return test_dls


def get_train_dls(train_data: FederationData, train_bs: int, benign_samples_per_device: Optional[int] = None, cuda: bool = False) -> List[DataLoader]:
    return [get_train_dl(client_train_data, train_bs, benign_samples_per_device=benign_samples_per_device, cuda=cuda)
            for client_train_data in train_data]


def get_val_dls(val_data: FederationData, test_bs: int, benign_samples_per_device: Optional[int] = None, cuda: bool = False) -> List[DataLoader]:
    return [get_val_dl(client_val_data, test_bs, benign_samples_per_device=benign_samples_per_device, cuda=cuda) for client_val_data in val_data]


def get_test_dls_dicts(local_test_data: FederationData, test_bs: int, benign_samples_per_device: Optional[int] = None,
                       attack_samples_per_device: Optional[int] = None, cuda: bool = False) -> List[Dict[str, DataLoader]]:
    return [get_test_dls_dict(client_test_data, test_bs, benign_samples_per_device=benign_samples_per_device,
                              attack_samples_per_device=attack_samples_per_device, cuda=cuda)
            for client_test_data in local_test_data]


def get_client_unsupervised_initial_splitting(client_data: ClientData, p_test: float, p_unused: float) -> Tuple[ClientData, ClientData]:
    # Separate the data of the clients between benign and attack
    client_benign_data = [{'benign': device_data['benign']} for device_data in client_data]
    client_attack_data = [{key: device_data[key] for key in device_data.keys() if key != 'benign'} for device_data in client_data]
    client_train_val, client_benign_test = split_client_data(client_benign_data, p_second_split=p_test, p_unused=p_unused)
    client_test = [{**device_benign_test, **device_attack_data}
                   for device_benign_test, device_attack_data in zip(client_benign_test, client_attack_data)]

    return client_train_val, client_test


def prepare_dataloaders(train_val_data: FederationData, local_test_data: FederationData, new_test_data: ClientData, params: SimpleNamespace)\
        -> Tuple[List[DataLoader], List[DataLoader], List[Dict[str, DataLoader]], Dict[str, DataLoader]]:
    # Split train data between actual train and the set that will be used to search the threshold
    train_data, threshold_data = split_clients_data(train_val_data, p_second_split=params.threshold_part, p_unused=0.0)

    p_train = params.p_train_val * (1. - params.threshold_part)
    p_threshold = params.p_train_val * params.threshold_part

    # Creating the dataloaders
    benign_samples_per_device, _ = get_benign_attack_samples_per_device(p_split=p_train,
                                                                        benign_prop=1., samples_per_device=params.samples_per_device)
    train_dls = get_train_dls(train_data, params.train_bs, benign_samples_per_device=benign_samples_per_device, cuda=params.cuda)

    benign_samples_per_device, _ = get_benign_attack_samples_per_device(p_split=p_threshold,
                                                                        benign_prop=1., samples_per_device=params.samples_per_device)
    threshold_dls = get_val_dls(threshold_data, params.test_bs, benign_samples_per_device=benign_samples_per_device, cuda=params.cuda)

    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_test, benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    local_test_dls_dicts = get_test_dls_dicts(local_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                                              attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    new_test_dls_dict = get_test_dls_dict(new_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                                          attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    return train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict
