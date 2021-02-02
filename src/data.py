from typing import Tuple, Dict, List, Union

import numpy as np
import pandas as pd
from context_printer import Color
from context_printer import ContextPrinter as Ctp

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


# Note that if the number of rows in a dataframe is not a multiple of the number of splits, some rows will be left out of all splits
# def get_splits(device_dataframes: dict, splits_benign: list, splits_attack: list) -> dict:
#     # Compute the indexes of the splits.
#     # For example for benign data, with splits_benign = [0.3, 0.7] and device_dataframes['benign'] of length 100,
#     # we would have indexes['benign'] = [0, 30, 100]
#     indexes = {'benign': [0] + list(np.cumsum([int(split_proportion * len(device_dataframes['benign'])) for split_proportion in splits_benign]))}
#     indexes.update({key: [0] + list(np.cumsum([int(split_proportion * len(device_dataframes[key])) for split_proportion in splits_attack]))
#                     for key in device_dataframes.keys() if key != 'benign'})
#
#     # For each key (benign, mirai_ack, ...) and for each split we take the corresponding rows of the corresponding dataframe
#     data_splits = {key: [torch.tensor(device_dataframes[key][indexes[key][split_id]:indexes[key][split_id + 1]]).float()
#                          for split_id in range(len(indexes[key]) - 1)]
#                    for key in device_dataframes.keys()}
#     return data_splits
