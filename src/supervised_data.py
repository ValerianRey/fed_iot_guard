from types import SimpleNamespace
from typing import List, Tuple, Optional, Set

import numpy as np
import torch
import torch.utils
import torch.utils.data
# noinspection PyProtectedMember
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import multiclass_labels, ClientData, FederationData, split_client_data, resample_array, get_benign_attack_samples_per_device


def get_target_tensor(key: str, arr: np.ndarray, multiclass: bool = False,
                      poisoning: Optional[str] = None, p_poison: Optional[float] = None) -> torch.Tensor:
    if multiclass:
        if poisoning is not None:
            raise NotImplementedError('Poisoning not implemented for multiclass data')
        return torch.full((arr.shape[0], 1), multiclass_labels[key])
    else:
        target = torch.full((arr.shape[0], 1), (0. if key == 'benign' else 1.))
        if poisoning is not None:
            if poisoning == 'all_labels_flipping' \
                    or (poisoning == 'benign_labels_flipping' and key == 'benign') \
                    or (poisoning == 'attack_labels_flipping' and key != 'benign'):
                if p_poison is None:
                    raise ValueError('p_poison should be indicated for label flipping attack')
                n_poisoned = int(p_poison * len(target))
                poisoned_indices = np.random.choice(len(target), n_poisoned, replace=False)
                # We turn 0 into 1 and 1 into 0 by subtracting 1 and raising to the power of 2
                target[poisoned_indices] = torch.pow(target[poisoned_indices] - 1., 2)

        return target


# Creates a dataset with the given client's data. If n_benign and n_attack are specified, up or down sampling will be used to have the right
# amount of that class of data. The data can also be poisoned if needed.
def get_dataset(data: ClientData, benign_samples_per_device: Optional[int] = None, attack_samples_per_device: Optional[int] = None,
                cuda: bool = False, multiclass: bool = False, poisoning: Optional[str] = None, p_poison: Optional[float] = None) -> Dataset:
    data_list, target_list = [], []
    resample = benign_samples_per_device is not None and attack_samples_per_device is not None

    for device_data in data:
        number_of_attacks = len(device_data.keys()) - 1
        n_samples_attack = attack_samples_per_device // 10
        if number_of_attacks == 5:
            n_samples_attack *= 2
        for key, arr in device_data.items():  # This will iterate over the benign splits, gafgyt splits and mirai splits (if applicable)
            if resample:
                if key == 'benign':
                    arr = resample_array(arr, benign_samples_per_device)
                else:
                    # We evenly divide the attack samples among the existing attacks on that device
                    arr = resample_array(arr, n_samples_attack)

            data_tensor = torch.tensor(arr).float()
            target_tensor = get_target_tensor(key, arr, multiclass=multiclass, poisoning=poisoning, p_poison=p_poison)
            if cuda:
                data_tensor = data_tensor.cuda()
                target_tensor = target_tensor.cuda()
            data_list.append(data_tensor)
            target_list.append(target_tensor)

    dataset = TensorDataset(torch.cat(data_list, dim=0), torch.cat(target_list, dim=0))
    return dataset


def get_train_dl(client_train_data: ClientData, train_bs: int, benign_samples_per_device: Optional[int] = None,
                 attack_samples_per_device: Optional[int] = None,
                 cuda: bool = False, multiclass: bool = False, poisoning: Optional[str] = None,
                 p_poison: Optional[float] = None) -> DataLoader:
    dataset_train = get_dataset(client_train_data, benign_samples_per_device=benign_samples_per_device,
                                attack_samples_per_device=attack_samples_per_device, cuda=cuda,
                                multiclass=multiclass, poisoning=poisoning, p_poison=p_poison)
    train_dl = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    return train_dl


def get_test_dl(client_test_data: ClientData, test_bs: int, benign_samples_per_device: Optional[int] = None,
                attack_samples_per_device: Optional[int] = None,
                cuda: bool = False, multiclass: bool = False) -> DataLoader:
    dataset_test = get_dataset(client_test_data, benign_samples_per_device=benign_samples_per_device,
                               attack_samples_per_device=attack_samples_per_device, cuda=cuda, multiclass=multiclass)
    test_dl = DataLoader(dataset_test, batch_size=test_bs)
    return test_dl


def get_train_dls(train_data: FederationData, train_bs: int, malicious_clients: Set[int],
                  benign_samples_per_device: Optional[int] = None, attack_samples_per_device: Optional[int] = None, cuda: bool = False,
                  multiclass: bool = False, poisoning: Optional[str] = None,
                  p_poison: Optional[float] = None) -> List[DataLoader]:
    train_dls = [get_train_dl(client_train_data, train_bs,
                              benign_samples_per_device=benign_samples_per_device, attack_samples_per_device=attack_samples_per_device,
                              cuda=cuda, multiclass=multiclass,
                              poisoning=(poisoning if client_id in malicious_clients else None),
                              p_poison=(p_poison if client_id in malicious_clients else None))
                 for client_id, client_train_data in enumerate(train_data)]
    return train_dls


def get_test_dls(test_data: FederationData, test_bs: int, benign_samples_per_device: Optional[int] = None,
                 attack_samples_per_device: Optional[int] = None,
                 cuda: bool = False, multiclass: bool = False) -> List[DataLoader]:
    test_dls = [get_test_dl(client_test_data, test_bs, benign_samples_per_device=benign_samples_per_device,
                            attack_samples_per_device=attack_samples_per_device, cuda=cuda, multiclass=multiclass)
                for client_test_data in test_data]
    return test_dls


def get_client_supervised_initial_splitting(client_data: ClientData, p_test: float, p_unused: float) -> Tuple[ClientData, ClientData]:
    client_train_val, client_test = split_client_data(client_data, p_second_split=p_test, p_unused=p_unused)
    return client_train_val, client_test


def prepare_dataloaders(train_data: FederationData, local_test_data: FederationData, new_test_data: ClientData, params: SimpleNamespace,
                        federated: bool = False) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    if federated:
        malicious_clients = params.malicious_clients
        poisoning = params.data_poisoning
        p_poison = params.p_poison
    else:
        malicious_clients = set()
        poisoning = None
        p_poison = None

    # Creating the dataloaders
    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_train_val,
                                                                                                benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    train_dls = get_train_dls(train_data, params.train_bs, malicious_clients=malicious_clients, benign_samples_per_device=benign_samples_per_device,
                              attack_samples_per_device=attack_samples_per_device, cuda=params.cuda, poisoning=poisoning, p_poison=p_poison)

    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_test, benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    local_test_dls = get_test_dls(local_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                                  attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    new_test_dl = get_test_dl(new_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                              attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    return train_dls, local_test_dls, new_test_dl
