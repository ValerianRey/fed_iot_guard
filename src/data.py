from typing import Tuple, Dict, List

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


def device_names(device_ids: List[int]) -> str:
    return ', '.join([all_devices[device_id] for device_id in device_ids])


def get_device_data(device_id: int) -> Dict[str, np.ndarray]:
    Ctp.print('[{}/{}] Data from '.format(device_id + 1, len(all_devices)) + all_devices[device_id])
    device = all_devices[device_id]
    device_data = {'benign': pd.read_csv(benign_paths[device]).to_numpy()}
    if device in mirai_devices:
        device_data.update({'mirai_' + attack: pd.read_csv(attack_paths[device]).to_numpy()
                            for attack, attack_paths in zip(mirai_attacks, mirai_paths)})

    device_data.update({'gafgyt_' + attack: pd.read_csv(attack_paths[device]).to_numpy()
                        for attack, attack_paths in zip(gafgyt_attacks, gafgyt_paths)})
    return device_data


def get_all_data() -> List[Dict[str, np.ndarray]]:
    Ctp.enter_section('Reading data', Color.YELLOW)
    data = [get_device_data(device_id) for device_id in range(len(all_devices))]
    Ctp.exit_section()
    return data


# Returns the clients' devices' data (first element of the tuple) and the test devices' data (second element of the tuple)
def get_configuration_split(all_data: List[Dict[str, np.ndarray]], clients_devices: List[List[int]], test_devices: List[int]) \
        -> Tuple[List[List[Dict[str, np.ndarray]]], List[Dict[str, np.ndarray]]]:

    clients_devices_data = [[all_data[device_id] for device_id in client_devices] for client_devices in clients_devices]
    test_devices_data = [all_data[device_id] for device_id in test_devices]
    return clients_devices_data, test_devices_data


# Turns a list of dicts into a single dict where all the arrays of each key are appended together
def get_client_data_combined(client_devices_data: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    combined_dict = {}
    for client_device_data in client_devices_data:
        for key, arr in client_device_data.items():
            print(arr.shape)
            if key in combined_dict.keys():
                combined_dict[key] = np.append(combined_dict[key], arr, axis=0)
            else:
                combined_dict[key] = arr

    return combined_dict


def split_clients_data(data: List[List[Dict[str, np.ndarray]]], p_test: float, p_unused: float) \
        -> Tuple[List[List[Dict[str, np.ndarray]]], List[List[Dict[str, np.ndarray]]]]:
    train_data, test_data = [], []
    for client_id, client_data in enumerate(data):
        client_train_data, client_test_data = split_client_data(client_data, p_test=p_test, p_unused=p_unused)
        train_data.append(client_train_data)
        test_data.append(client_test_data)

    return train_data, test_data


def split_client_data(data: List[Dict[str, np.ndarray]], p_test: float, p_unused: float) \
        -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:

    p_train = 1 - p_test - p_unused
    train_data, test_data = [], []
    for device_id, device_data in enumerate(data):
        train_data.append({})
        test_data.append({})
        for key, array in device_data.items():
            indexes = [0] + list(np.cumsum((len(array) * np.array([p_train, p_unused, p_test])).astype(int)))
            train_data[device_id][key] = array[indexes[0]:indexes[1]]
            test_data[device_id][key] = array[indexes[2]:indexes[3]]

    return train_data, test_data


def split_client_data_current_fold(train_val_data: List[Dict[str, np.ndarray]], n_folds: int, fold: int) \
        -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:

    kf = KFold(n_splits=n_folds)
    train_data, val_data = [], []
    for device_id, device_data in enumerate(train_val_data):
        train_data.append({})
        val_data.append({})
        for key, array in device_data.items():
            train_index, val_index = list(kf.split(array))[fold]
            train_data[device_id][key] = array[train_index]
            val_data[device_id][key] = array[val_index]

    return train_data, val_data


def split_clients_data_current_fold(train_val_data: List[List[Dict[str, np.ndarray]]], n_folds: int, fold: int) \
        -> Tuple[List[List[Dict[str, np.ndarray]]], List[List[Dict[str, np.ndarray]]]]:

    train_data, val_data = [], []

    for client_id, client_data in enumerate(train_val_data):
        client_train_data, client_val_data = split_client_data_current_fold(client_data, n_folds=n_folds, fold=fold)
        train_data.append(client_train_data)
        val_data.append(client_val_data)

    return train_data, val_data
