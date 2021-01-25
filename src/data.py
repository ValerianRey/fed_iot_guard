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


def get_sub_div(train_data, normalization):
    if normalization == '0-mean 1-var':
        sub = train_data.mean(dim=0)
        div = train_data.std(dim=0)
    elif normalization == 'min-max':
        sub = train_data.min(dim=0)[0]
        div = train_data.max(dim=0)[0] - sub
    else:
        sub = 0.
        div = 1.
    return sub, div


def get_dataframes(devices):
    benign_dataframes = [pd.read_csv(benign_paths[device]) for device in devices]
    mirai_dataframes = [[pd.read_csv(attack_paths[device]) for device in devices if device in mirai_devices]
                        for attack_paths in mirai_paths]
    gafgyt_dataframes = [[pd.read_csv(attack_paths[device]) for device in devices] for attack_paths in gafgyt_paths]
    return benign_dataframes, mirai_dataframes, gafgyt_dataframes


def get_splits(devices, splits_benign=1, splits_attack=1, chronological=True):
    benign_dataframes, mirai_dataframes, gafgyt_dataframes = get_dataframes(devices)

    if chronological:
        benign_indexes = [[slice(split * (len(df.values) // splits_benign),
                                 (split + 1) * (len(df.values) // splits_benign))
                           for df in benign_dataframes]
                          for split in range(splits_benign)]

        mirai_indexes = [[[slice(split * (len(df.values) // splits_attack),
                                 (split + 1) * (len(df.values) // splits_attack))
                           for df in mirai_dataframes[attack_id]]
                          for attack_id in range(len(mirai_attacks))]
                         for split in range(splits_attack)]

        gafgyt_indexes = [[[slice(split * (len(df.values) // splits_attack),
                                  (split + 1) * (len(df.values) // splits_attack))
                            for df in gafgyt_dataframes[attack_id]]
                           for attack_id in range(len(gafgyt_attacks))]
                          for split in range(splits_attack)]

    else:
        raise NotImplementedError()

    benign_data_splits = [torch.cat([torch.tensor(df.values[benign_indexes[split_id][device_id]]).float()
                                     for (device_id, df) in enumerate(benign_dataframes)], dim=0)
                          for split_id in range(splits_benign)]

    mirai_sensitive_devices = len(mirai_dataframes[0])
    if mirai_sensitive_devices > 0:
        mirai_data_splits = [[torch.cat([torch.tensor(df.values[mirai_indexes[split_id][attack_id][device_id]]).float()
                                         for (device_id, df) in enumerate(mirai_dataframes[attack_id])], dim=0)
                              for attack_id in range(len(mirai_attacks))]
                             for split_id in range(splits_attack)]
    else:
        mirai_data_splits = [None for _ in range(splits_attack)]

    gafgyt_data_splits = [[torch.cat([torch.tensor(df.values[gafgyt_indexes[split_id][attack_id][device_id]]).float()
                                      for (device_id, df) in enumerate(gafgyt_dataframes[attack_id])], dim=0)
                           for attack_id in range(len(gafgyt_attacks))]
                          for split_id in range(splits_attack)]

    return benign_data_splits, mirai_data_splits, gafgyt_data_splits


def get_data(devices, supervised, normalization='0-mean 1-var'):
    if supervised:
        splits_benign = 2
        splits_attack = 2
    else:
        splits_benign = 3
        splits_attack = 1

    benign_data_splits, mirai_data_splits, gafgyt_data_splits = get_splits(devices,
                                                                           splits_benign=splits_benign,
                                                                           splits_attack=splits_attack)

    # TODO: this is the correct way to normalize, however due to class imbalance (too much attack data compared to benign data) and to the fact
    #  that we use a baseline binary cross entropy loss function (thus not correcting the class imbalance) it ends up being better to normalize
    #  using only benign data for now (in order to give an advantage to benign data for the learned classifier). This is not cheating because
    #  such data is in the training set, but it's quite unusual and it should be changed later on once we address the class imbalance problem
    # if supervised:
    #     # When using the supervised more we want to normalize based on the whole training set (benign + attack)
    #     if mirai_data_splits[0] is not None:
    #         print('mirai data not none')
    #         data_to_compute_normalization = torch.cat([benign_data_splits[0]] + mirai_data_splits[0] + gafgyt_data_splits[0], dim=0)
    #     else:
    #         print('mirai data none')
    #         data_to_compute_normalization = torch.cat([benign_data_splits[0]] + gafgyt_data_splits[0], dim=0)
    # else:
    #     data_to_compute_normalization = benign_data_splits[0]

    data_to_compute_normalization = benign_data_splits[0]

    sub, div = get_sub_div(data_to_compute_normalization, normalization)

    benign_data_splits = [(split - sub) / div for split in benign_data_splits]

    mirai_data_splits = [[(attack_data - sub) / div for attack_data in split] if split is not None else None
                         for split in mirai_data_splits]

    gafgyt_data_splits = [[(attack_data - sub) / div for attack_data in split]
                          for split in gafgyt_data_splits]

    return tuple(benign_data_splits + mirai_data_splits + gafgyt_data_splits)


def get_classifier_datasets(devices, normalization='0-mean 1-var'):
    benign_data_train, benign_data_test, mirai_data_train, mirai_data_test, gafgyt_data_train, gafgyt_data_test = \
        get_data(devices, supervised=True, normalization=normalization)

    # Creating datasets
    if mirai_data_train is not None and mirai_data_test is not None:
        data_train = torch.cat([benign_data_train] + mirai_data_train + gafgyt_data_train, dim=0)
        data_test = torch.cat([benign_data_test] + mirai_data_test + gafgyt_data_test, dim=0)
        mirai_samples_train = sum([len(attack_data) for attack_data in mirai_data_train])
        mirai_samples_test = sum([len(attack_data) for attack_data in mirai_data_test])
    else:
        data_train = torch.cat([benign_data_train] + gafgyt_data_train, dim=0)
        data_test = torch.cat([benign_data_test] + gafgyt_data_test, dim=0)
        mirai_samples_train = 0
        mirai_samples_test = 0

    gafgyt_samples_train = sum([len(attack_data) for attack_data in gafgyt_data_train])
    gafgyt_samples_test = sum([len(attack_data) for attack_data in gafgyt_data_test])

    attack_samples_train = mirai_samples_train + gafgyt_samples_train
    attack_samples_test = mirai_samples_test + gafgyt_samples_test

    targets_train = torch.cat([torch.zeros(len(benign_data_train), 1)] + [torch.ones(attack_samples_train, 1)], dim=0)
    targets_test = torch.cat([torch.zeros(len(benign_data_test), 1)] + [torch.ones(attack_samples_test, 1)], dim=0)

    dataset_train = torch.utils.data.TensorDataset(data_train, targets_train)
    dataset_test = torch.utils.data.TensorDataset(data_test, targets_test)

    return dataset_train, dataset_test


def get_autoencoder_datasets(devices, normalization='0-mean 1-var'):
    benign_data_train, benign_data_opt, benign_data_test, mirai_data, gafgyt_data = \
        get_data(devices, supervised=False, normalization=normalization)

    dataset_benign_train = torch.utils.data.TensorDataset(benign_data_train)
    dataset_benign_opt = torch.utils.data.TensorDataset(benign_data_opt)
    dataset_benign_test = torch.utils.data.TensorDataset(benign_data_test)

    if mirai_data is not None:
        datasets_mirai = [torch.utils.data.TensorDataset(attack_data) for attack_data in mirai_data]
    else:
        datasets_mirai = None

    datasets_gafgyt = [torch.utils.data.TensorDataset(attack_data) for attack_data in gafgyt_data]

    return dataset_benign_train, dataset_benign_opt, dataset_benign_test, datasets_mirai, datasets_gafgyt


def get_devices_ids(args):
    device_ids = []
    for client_devices in args.clients_devices:
        for device_id in client_devices:
            if device_id not in device_ids:
                device_ids.append(device_id)

    for device_id in args.test_devices:
        if device_id not in device_ids:
            device_ids.append(device_id)
    return device_ids


def get_autoencoder_dataloaders(args, devices_list, ctp: ContextPrinter, color=Color.NONE):
    ctp.print('Reading data', color=color, bold=True)
    ctp.add_bar(color)

    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt = [], [], [], [], []

    for i, devices in enumerate(devices_list):
        ctp.print('[{}/{}] Data from '.format(i + 1, len(devices_list)), end='')
        if type(devices) == list:
            print('{} devices: '.format(len(devices)) + ', '.join(devices))
            dataset = get_autoencoder_datasets(devices, args.normalization)
        else:
            print(devices)
            dataset = get_autoencoder_datasets([devices], args.normalization)

        dataloaders_train.append(torch.utils.data.DataLoader(dataset[0], batch_size=args.train_bs, shuffle=True))
        dataloaders_opt.append(torch.utils.data.DataLoader(dataset[1], batch_size=args.test_bs))
        dataloaders_benign_test.append(torch.utils.data.DataLoader(dataset[2], batch_size=args.test_bs))
        if dataset[3] is not None:
            dataloaders_mirai.append([torch.utils.data.DataLoader(dataset[3][i], batch_size=args.test_bs) for i in range(len(mirai_attacks))])
        else:
            dataloaders_mirai.append(None)
        dataloaders_gafgyt.append([torch.utils.data.DataLoader(dataset[4][i], batch_size=args.test_bs) for i in range(len(gafgyt_attacks))])

    ctp.remove_header()
    return dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt


def get_autoencoder_dataloaders_v2(args, ctp: ContextPrinter, color=Color.NONE):
    ctp.print('Reading data', color=color, bold=True)
    ctp.add_bar(color)

    # Step 1: construct the list of all the devices for which we need to read data
    device_ids = get_devices_ids(args)

    # Step 2: load the data
    datasets = {idx: None for idx in range(len(all_devices))}
    for i, device_id in enumerate(device_ids):
        ctp.print('[{}/{}] Data from '.format(i + 1, len(device_ids)) + all_devices[device_id])
        datasets[device_id] = get_autoencoder_datasets([device_id], args.normalization)

    # Step 3: create the dataloaders
    clients_dataloaders_train, clients_dataloaders_opt, clients_dataloaders_benign_test, clients_dataloaders_mirai, clients_dataloaders_gafgyt = \
        [], [], [], [], []

    for client_devices in args.clients_devices:
        combined_dataset_train = torch.utils.data.ConcatDataset([datasets[device_id][0] for device_id in client_devices])
        clients_dataloaders_train.append(torch.utils.data.DataLoader(combined_dataset_train, batch_size=args.train_bs, shuffle=True))

        combined_dataset_opt = torch.utils.data.ConcatDataset([datasets[device_id][1] for device_id in client_devices])
        clients_dataloaders_opt.append(torch.utils.data.DataLoader(combined_dataset_opt, batch_size=args.test_bs))

        combined_dataset_benign_test = torch.utils.data.ConcatDataset([datasets[device_id][2] for device_id in client_devices])
        clients_dataloaders_benign_test.append(torch.utils.data.DataLoader(combined_dataset_benign_test, batch_size=args.test_bs))

        datasets_mirai = [datasets[device_id][3] for device_id in client_devices if datasets[device_id][3] is not None]
        if len(datasets_mirai) >= 1:
            combined_datasets_mirai = [torch.utils.data.ConcatDataset(datasets_mirai[device_id][i] for device_id in client_devices)
                                       for i in range(len(mirai_attacks))]
            clients_dataloaders_mirai.append([torch.utils.data.DataLoader(combined_datasets_mirai[i], batch_size=args.test_bs)
                                              for i in range(len(mirai_attacks))])
        else:
            clients_dataloaders_mirai.append(None)

        datasets_gafgyt = [datasets[device_id][4] for device_id in client_devices]
        combined_datasets_gafgyt = [torch.utils.data.ConcatDataset(datasets_gafgyt[device_id][i] for device_id in client_devices)
                                    for i in range(len(gafgyt_attacks))]
        clients_dataloaders_gafgyt.append([torch.utils.data.DataLoader(combined_datasets_gafgyt[i], batch_size=args.test_bs)
                                           for i in range(len(gafgyt_attacks))])

    combined_dataset_benign_test = torch.utils.data.ConcatDataset([datasets[device_id][2] for device_id in args.test_devices])
    new_dataloader_benign_test = torch.utils.data.DataLoader(combined_dataset_benign_test, batch_size=args.test_bs)

    datasets_mirai = [datasets[device_id][3] for device_id in args.test_devices if datasets[device_id][3] is not None]
    if len(datasets_mirai) >= 1:
        combined_datasets_mirai = [torch.utils.data.ConcatDataset(datasets_mirai[device_id][i] for device_id in args.test_devices)
                                   for i in range(len(mirai_attacks))]
        new_dataloaders_mirai = [torch.utils.data.DataLoader(combined_datasets_mirai[i], batch_size=args.test_bs)
                                 for i in range(len(mirai_attacks))]
    else:
        new_dataloaders_mirai = None

    datasets_gafgyt = [datasets[device_id][4] for device_id in args.test_devices]
    combined_datasets_gafgyt = [torch.utils.data.ConcatDataset(datasets_gafgyt[device_id][i] for device_id in args.test_devices)
                                for i in range(len(gafgyt_attacks))]
    new_dataloaders_gafgyt = [torch.utils.data.DataLoader(combined_datasets_gafgyt[i], batch_size=args.test_bs)
                              for i in range(len(gafgyt_attacks))]

    ctp.remove_header()
    return clients_dataloaders_train, clients_dataloaders_opt, clients_dataloaders_benign_test, \
           clients_dataloaders_mirai, clients_dataloaders_gafgyt, new_dataloader_benign_test, new_dataloaders_mirai, new_dataloaders_gafgyt


# args.clients_devices should be a list of lists of devices and args.test_devices should be a list of devices
def get_classifier_dataloaders(args, ctp: ContextPrinter, color=Color.NONE):
    ctp.print('Reading data', color=color, bold=True)
    ctp.add_bar(color)

    # Step 1: construct the list of all the devices for which we need to read data
    device_ids = get_devices_ids(args)

    # Step 2: load the data
    datasets = {idx: None for idx in range(len(all_devices))}
    for i, device_id in enumerate(device_ids):
        ctp.print('[{}/{}] Data from '.format(i + 1, len(device_ids)) + all_devices[device_id])
        datasets[device_id] = get_classifier_datasets([all_devices[device_id]], normalization=args.normalization)

    # Step 3: create the dataloaders
    clients_dataloaders_train, clients_dataloaders_test = [], []
    for client_devices in args.clients_devices:
        client_datasets = [datasets[device_id] for device_id in client_devices]
        combined_dataset_train = torch.utils.data.ConcatDataset([dataset[0] for dataset in client_datasets])
        combined_dataset_test = torch.utils.data.ConcatDataset([dataset[1] for dataset in client_datasets])
        clients_dataloaders_train.append(torch.utils.data.DataLoader(combined_dataset_train, batch_size=args.train_bs, shuffle=True))
        clients_dataloaders_test.append(torch.utils.data.DataLoader(combined_dataset_test, batch_size=args.test_bs))

    combined_dataset_test = torch.utils.data.ConcatDataset([dataset[1] for dataset in [datasets[device_id] for device_id in args.test_devices]])
    new_dataloader_test = torch.utils.data.DataLoader(combined_dataset_test, batch_size=args.test_bs)

    ctp.remove_header()
    return clients_dataloaders_train, clients_dataloaders_test, new_dataloader_test
