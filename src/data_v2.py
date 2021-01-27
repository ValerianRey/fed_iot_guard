from types import SimpleNamespace

import pandas as pd
import torch
import torch.utils.data

from print_util import ContextPrinter, Color

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


def test():
    dfs1 = get_dataframes(1)
    dfs2 = get_dataframes(2)
    args = SimpleNamespace(**{'clients_devices': [[0], [1], [2], [3], [4], [5], [6], [7]], 'test_devices': [8],
                              'normalization': '0-mean 1-var', 'train_bs': 64, 'test_bs': 4096})
    ctp = ContextPrinter()
    clients_dl_train, clients_dl_test, new_dl_test = get_classifier_dataloaders(args, ctp=ctp)
    print(len(clients_dl_train))
    print(len(clients_dl_train))
    print(len(new_dl_test))


def device_names(device_ids):
    return ', '.join([all_devices[device_id] for device_id in device_ids])


def get_sub_div(data, normalization):
    if normalization == '0-mean 1-var':
        sub = data.mean(dim=0)
        div = data.std(dim=0)
    elif normalization == 'min-max':
        sub = data.min(dim=0)[0]
        div = data.max(dim=0)[0] - sub
    elif normalization == 'none':
        sub = torch.zeros(data.shape[1])
        div = torch.ones(data.shape[1])
    else:
        raise NotImplementedError

    return sub, div


def get_sub_div_dl(dataloader: torch.utils.data.DataLoader, normalization):
    return get_sub_div(dataloader.dataset[:], normalization)


def get_dataframes(device_id: int):
    device = all_devices[device_id]
    dataframes = {'benign': pd.read_csv(benign_paths[device])}
    if device in mirai_devices:
        dataframes.update({'mirai_' + attack: pd.read_csv(attack_paths[device]) for attack, attack_paths in zip(mirai_attacks, mirai_paths)})

    dataframes.update({'gafgyt_' + attack: pd.read_csv(attack_paths[device]) for attack, attack_paths in zip(gafgyt_attacks, gafgyt_paths)})
    return dataframes


def get_splits(device_dataframes: dict, splits_benign=1, splits_attack=1, chronological=True):
    # Note that if the number of rows in a dataframe is not a multiple of the number of splits, some rows will be left out of all splits
    def get_slice(split, length, num_splits):
        return slice(split * (length // num_splits), (split + 1) * (length // num_splits))

    if chronological:
        benign_indexes = [get_slice(split, len(device_dataframes['benign'].values), splits_benign) for split in range(splits_benign)]

        attack_indexes = [{key: get_slice(split, len(df.values), splits_attack) for key, df in device_dataframes.items()}
                          for split in range(splits_attack)]
    else:
        raise NotImplementedError()

    # Get the benign dataframes and remove the benign key from the dict: we now only have attacks left in the dict
    benign_df = device_dataframes.pop('benign')
    data_splits = {'benign': [torch.tensor(benign_df.values[benign_indexes[split_id]]).float() for split_id in range(splits_benign)]}

    for attack, attack_df in device_dataframes.items():
        data_splits.update({attack: [torch.tensor(attack_df.values[attack_indexes[split][attack]]).float() for split in range(splits_attack)]})

    return data_splits


def get_classifier_datasets(devices_dataframes: list):
    train_data = []
    test_data = []
    train_targets = []
    test_targets = []
    for df in devices_dataframes:
        data_splits = get_splits(df, splits_benign=2, splits_attack=2, chronological=True)
        for key, splits in data_splits.items():  # This will iterate over the benign splits, gafgyt splits and mirai splits (if applicable)
            train_data.append(splits[0])
            test_data.append(splits[1])
            train_targets.append(torch.full_like(splits[0], (0 if key == 'benign' else 1)))
            test_targets.append(torch.full_like(splits[0], (0 if key == 'benign' else 1)))

    dataset_train = torch.utils.data.TensorDataset(torch.cat(train_data, dim=0), torch.cat(train_targets, dim=0))
    dataset_test = torch.utils.data.TensorDataset(torch.cat(test_data, dim=0), torch.cat(test_targets, dim=0))

    return dataset_train, dataset_test


def get_devices_ids(args):
    device_ids = []
    for client_device_ids in args.clients_devices:
        for device_id in client_device_ids:
            if device_id not in device_ids:
                device_ids.append(device_id)

    for device_id in args.test_devices:
        if device_id not in device_ids:
            device_ids.append(device_id)
    return device_ids


def get_client_classifier_dataloaders(args, client_device_ids, device_id_to_dataframes):
    devices_dataframes = [device_id_to_dataframes[device_id] for device_id in client_device_ids]
    dataset_train, dataset_test = get_classifier_datasets(devices_dataframes)
    client_dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_bs, shuffle=True)
    client_dl_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_bs)
    return client_dl_train, client_dl_test


# args.clients_devices should be a list of lists of devices and args.test_devices should be a list of devices
def get_classifier_dataloaders(args, ctp: ContextPrinter, color=Color.NONE):
    ctp.print('Reading data', color=color, bold=True)
    ctp.add_bar(color)

    # Step 1: construct the list of all the devices for which we need to read data
    device_ids = get_devices_ids(args)

    # Step 2: load the data
    device_id_to_dataframes = {}
    for i, device_id in enumerate(device_ids):
        ctp.print('[{}/{}] Data from '.format(i + 1, len(device_ids)) + all_devices[device_id])
        device_id_to_dataframes.update({device_id: get_dataframes(device_id)})
    print(device_id_to_dataframes)

    # Step 3: create the datasets and the dataloaders of the clients: 1 train and 1 test per client
    clients_dl_train, clients_dl_test = [], []
    for client_device_ids in args.clients_devices:
        client_dl_train, client_dl_test = get_client_classifier_dataloaders(args, client_device_ids, device_id_to_dataframes)
        clients_dl_train.append(client_dl_train)
        clients_dl_test.append(client_dl_test)

    # Step 4: create the dataset and the dataloader of the new devices (test only)
    _, new_dl_test = get_client_classifier_dataloaders(args, args.test_devices, device_id_to_dataframes)

    print("Number of clients: " + repr(len(args.clients_devices)))
    print("Number of client's train dataloaders: " + repr(len(clients_dl_train)))
    print("Number of client's test dataloaders: " + repr(len(clients_dl_test)))

    ctp.remove_header()
    return clients_dl_train, clients_dl_test, new_dl_test


if __name__ == '__main__':
    test()
