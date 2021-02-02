from typing import Tuple, Dict, List, Union

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


def device_names(device_ids: List[int]) -> str:
    return ', '.join([all_devices[device_id] for device_id in device_ids])


def get_device_data(device_id: int) -> Dict[str, np.array]:
    Ctp.print('[{}/{}] Data from '.format(device_id + 1, len(all_devices)) + all_devices[device_id])
    device = all_devices[device_id]
    device_data = {'benign': pd.read_csv(benign_paths[device]).to_numpy()}
    if device in mirai_devices:
        device_data.update({'mirai_' + attack: pd.read_csv(attack_paths[device]).to_numpy()
                            for attack, attack_paths in zip(mirai_attacks, mirai_paths)})

    device_data.update({'gafgyt_' + attack: pd.read_csv(attack_paths[device]).to_numpy()
                        for attack, attack_paths in zip(gafgyt_attacks, gafgyt_paths)})
    return device_data


def get_all_data(color: Union[Color, str] = Color.NONE) -> List[Dict[str, np.array]]:
    Ctp.enter_section('Reading data', color)
    data = [get_device_data(device_id) for device_id in range(len(all_devices))]
    Ctp.exit_section()
    return data


def split_data(data: List[Dict[str, np.array]], p_test: float, p_unused: float) -> Tuple[List[Dict[str, np.array]], List[Dict[str, np.array]]]:
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


def split_data_current_fold(train_val_data: List[Dict[str, np.array]], n_folds: int, fold: int) \
        -> Tuple[List[Dict[str, np.array]], List[Dict[str, np.array]]]:

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



