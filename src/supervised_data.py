from typing import List, Tuple, Optional, Set

import numpy as np
import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import multiclass_labels, ClientData, FederationData, split_client_data, compute_alphas_resampling, resample_array


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


def get_dataset(data: ClientData, sampling: Optional[str] = None, p_benign: Optional[float] = None, cuda: bool = False,
                multiclass: bool = False, poisoning: Optional[str] = None, p_poison: Optional[float] = None) -> Dataset:
    data_list, target_list = [], []

    for device_data in data:
        benign_samples = sum([len(arr) for key, arr in device_data.items() if key == 'benign'])
        attack_samples = sum([len(arr) for key, arr in device_data.items() if key != 'benign'])
        alpha_benign, alpha_attack = compute_alphas_resampling(benign_samples, attack_samples, sampling, p_benign, verbose=True)

        for key, arr in device_data.items():  # This will iterate over the benign splits, gafgyt splits and mirai splits (if applicable)
            if key == 'benign':
                arr = resample_array(arr, alpha_benign)
            else:
                arr = resample_array(arr, alpha_attack)

            data_tensor = torch.tensor(arr).float()
            target_tensor = get_target_tensor(key, arr, multiclass=multiclass, poisoning=poisoning, p_poison=p_poison)
            if cuda:
                data_tensor = data_tensor.cuda()
                target_tensor = target_tensor.cuda()
            data_list.append(data_tensor)
            target_list.append(target_tensor)

    dataset = TensorDataset(torch.cat(data_list, dim=0), torch.cat(target_list, dim=0))
    return dataset


def get_train_dl(client_train_data: ClientData, train_bs: int, sampling: Optional[str] = None, p_benign: Optional[float] = None,
                 cuda: bool = False, multiclass: bool = False, poisoning: Optional[str] = None,
                 p_poison: Optional[float] = None) -> DataLoader:
    dataset_train = get_dataset(client_train_data, sampling=sampling, p_benign=p_benign, cuda=cuda,
                                multiclass=multiclass, poisoning=poisoning, p_poison=p_poison)
    train_dl = DataLoader(dataset_train, batch_size=train_bs, shuffle=True)
    return train_dl


def get_test_dl(client_test_data: ClientData, test_bs: int, sampling: Optional[str] = None, p_benign: Optional[float] = None,
                cuda: bool = False, multiclass: bool = False) -> DataLoader:
    dataset_test = get_dataset(client_test_data, sampling=sampling, p_benign=p_benign, cuda=cuda, multiclass=multiclass)
    test_dl = DataLoader(dataset_test, batch_size=test_bs)
    return test_dl


def get_train_test_dls(train_data: FederationData, test_data: FederationData, train_bs: int, test_bs: int, malicious_clients: Set[int],
                       sampling: Optional[str] = None, p_benign: Optional[float] = None, cuda: bool = False, multiclass: bool = False,
                       poisoning: Optional[str] = None, p_poison: Optional[float] = None) -> Tuple[List[DataLoader], List[DataLoader]]:

    train_dls = [get_train_dl(client_train_data, train_bs,
                              sampling=sampling, p_benign=p_benign, cuda=cuda, multiclass=multiclass,
                              poisoning=(poisoning if client_id in malicious_clients else None),
                              p_poison=(p_poison if client_id in malicious_clients else None))
                 for client_id, client_train_data in enumerate(train_data)]

    test_dls = [get_test_dl(client_test_data, test_bs, sampling=None, p_benign=None, cuda=cuda, multiclass=multiclass)
                for client_test_data in test_data]
    return train_dls, test_dls


def get_client_supervised_initial_splitting(client_data: ClientData, p_test: float, p_unused: float) -> Tuple[ClientData, ClientData]:
    client_train_val, client_test = split_client_data(client_data, p_test=p_test, p_unused=p_unused)
    return client_train_val, client_test
