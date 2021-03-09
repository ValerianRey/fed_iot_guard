from typing import Tuple, Dict, List, Callable

import numpy as np
import pandas as pd
from context_printer import Color
from context_printer import ContextPrinter as Ctp
from sklearn.model_selection import KFold

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

multiclass_labels = {**{'benign': 0.},
                     **{'mirai_' + attack: float(i+1) for i, attack in enumerate(mirai_attacks)},
                     **{'gafgyt_' + attack: float(i+6) for i, attack in enumerate(gafgyt_attacks)}}

DeviceData = Dict[str, np.ndarray]
ClientData = List[DeviceData]
FederationData = List[ClientData]


def device_names(device_ids: List[int]) -> str:
    return ', '.join([all_devices[device_id] for device_id in device_ids])


def read_device_data(device_id: int) -> DeviceData:
    Ctp.print('[{}/{}] Data from '.format(device_id + 1, len(all_devices)) + all_devices[device_id])
    device = all_devices[device_id]
    device_data = {'benign': pd.read_csv(benign_paths[device]).to_numpy()}
    if device in mirai_devices:
        device_data.update({'mirai_' + attack: pd.read_csv(attack_paths[device]).to_numpy()
                            for attack, attack_paths in zip(mirai_attacks, mirai_paths)})

    device_data.update({'gafgyt_' + attack: pd.read_csv(attack_paths[device]).to_numpy()
                        for attack, attack_paths in zip(gafgyt_attacks, gafgyt_paths)})
    return device_data


def read_all_data() -> List[DeviceData]:
    Ctp.enter_section('Reading data', Color.YELLOW)
    data = [read_device_data(device_id) for device_id in range(len(all_devices))]
    Ctp.exit_section()
    return data


def get_client_data(all_data: List[DeviceData], client_devices: List[int]) -> ClientData:
    return [all_data[device_id] for device_id in client_devices]


# Returns the clients' devices' data (first element of the tuple) and the test devices' data (second element of the tuple)
def get_configuration_data(all_data: List[DeviceData], clients_devices: List[List[int]], test_devices: List[int]) \
        -> Tuple[FederationData, ClientData]:

    clients_devices_data = [get_client_data(all_data, client_devices) for client_devices in clients_devices]
    test_devices_data = get_client_data(all_data, test_devices)
    return clients_devices_data, test_devices_data


def split_clients_data(data: FederationData, p_second_split: float, p_unused: float) -> Tuple[FederationData, FederationData]:
    train_data, test_data = [], []
    for client_data in data:
        client_train_data, client_test_data = split_client_data(client_data, p_second_split=p_second_split, p_unused=p_unused)
        train_data.append(client_train_data)
        test_data.append(client_test_data)

    return train_data, test_data


def split_client_data(data: ClientData, p_second_split: float, p_unused: float) -> Tuple[ClientData, ClientData]:
    p_first_split = 1 - p_second_split - p_unused
    train_data, test_data = [], []
    for device_id, device_data in enumerate(data):
        train_data.append({})
        test_data.append({})
        for key, array in device_data.items():
            indexes = [0] + list(np.cumsum((len(array) * np.array([p_first_split, p_unused, p_second_split])).astype(int)))
            train_data[device_id][key] = array[indexes[0]:indexes[1]]
            test_data[device_id][key] = array[indexes[2]:indexes[3]]

    return train_data, test_data


def split_client_data_current_fold(train_val_data: ClientData, n_splits: int, fold: int) \
        -> Tuple[ClientData, ClientData]:

    kf = KFold(n_splits=n_splits)
    train_data, val_data = [], []
    for device_id, device_data in enumerate(train_val_data):
        train_data.append({})
        val_data.append({})
        for key, array in device_data.items():
            train_index, val_index = list(kf.split(array))[fold]
            train_data[device_id][key] = array[train_index]
            val_data[device_id][key] = array[val_index]

    return train_data, val_data


def get_initial_splitting(splitting_function: Callable, clients_data: FederationData, p_test: float, p_unused: float) \
        -> Tuple[FederationData, FederationData]:
    clients_train_val, clients_test = [], []
    for client_data in clients_data:
        client_train_val, client_test = splitting_function(client_data, p_test=p_test, p_unused=p_unused)
        clients_train_val.append(client_train_val)
        clients_test.append(client_test)

    return clients_train_val, clients_test


def get_benign_attack_samples_per_device(p_split: float, benign_prop: float, samples_per_device: int) -> Tuple[int, int]:
    if benign_prop is None or samples_per_device is None:
        benign_samples_per_device, attack_samples_per_device = None, None
    else:
        benign_samples_per_device = int(round(samples_per_device * p_split * benign_prop, 5))
        attack_samples_per_device = int(round(samples_per_device * p_split * (1. - benign_prop), 5))

    return benign_samples_per_device, attack_samples_per_device


# Select n_samples rows from a numpy array, using either upsampling or downsampling.
def resample_array(arr: np.ndarray, n_samples: int) -> np.ndarray:
    # Compute the proportion between desired number of samples and input array's length
    alpha = n_samples / len(arr)

    # Repeat the original array as many times as possible (integer number of times)
    repeats = int(alpha)
    repeated_arr = arr.repeat(repeats, axis=0)

    # Sample randomly without replacement the remaining samples
    n_random_samples = n_samples - len(repeated_arr)
    all_indexes = np.arange(len(arr))
    np.random.seed(0)  # Fix the seed so that the resampling is not random (to have more meaningful results)
    random_arr = arr[np.random.choice(all_indexes, n_random_samples, replace=False)]
    np.random.seed(None)  # Reset the seed to a pseudo-random value so that the rest of the program is unaffected by the fixing of the seed
    result = np.append(repeated_arr, random_arr, axis=0)

    assert(len(result) == n_samples)

    return result
