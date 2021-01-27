from copy import deepcopy

from anomaly_detection_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds
from architectures import SimpleAutoencoder
from data import get_unsupervised_dataloaders, device_names
from federated_util import federated_averaging
from print_util import Color, ContextPrinter


# TODO: add normalization to the autoencoder model
def local_autoencoders(args):
    ctp = ContextPrinter()
    n_clients = len(args.clients_devices)
    if n_clients == 1:
        ctp.print('\n\t\t\t\t\tSINGLE CLASSIFIER\n', bold=True)
    else:
        ctp.print('\n\t\t\t\t\tMULTIPLE CLASSIFIERS\n', bold=True)

    # Initialization of the models
    models = [SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers) for _ in range(n_clients)]

    # Loading the data and creating the dataloaders
    clients_dl_train, clients_dl_opt, clients_dls_test, new_dls_test = get_unsupervised_dataloaders(args, ctp=ctp, color=Color.YELLOW)
    ctp.print('\n')

    # Local training of the autoencoder
    multitrain_autoencoders(trains=zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                       for i, client_devices in enumerate(args.clients_devices)], clients_dl_train, models),
                            args=args, ctp=ctp, main_title='Training the clients', color=Color.GREEN)
    ctp.print('\n')

    # Computation of the thresholds
    thresholds = compute_thresholds(opts=zip(['Computing threshold for client {} on: '.format(i + 1) + device_names(client_devices)
                                               for i, client_devices in enumerate(args.clients_devices)], clients_dl_opt, models),
                                    ctp=ctp, main_title='Computing the thresholds', color=Color.RED)
    ctp.print('\n')

    # Local testing of each autoencoder
    multitest_autoencoders(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(client_devices)
                                     for i, client_devices in enumerate(args.clients_devices)], clients_dls_test, models, thresholds),
                           ctp=ctp, main_title='Testing the clients on their own devices', color=Color.BLUE)
    ctp.print('\n')

    # New devices testing
    multitest_autoencoders(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(args.test_devices) for i in range(n_clients)],
                                     [new_dls_test for _ in range(n_clients)], models),
                           ctp=ctp, main_title='Testing the clients on the new devices: ' + device_names(args.test_devices), color=Color.DARKCYAN)


# TODO: update with the new dataloading process and with the new experiments
def federated_autoencoders(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tFEDERATED AUTOENCODERS\n', bold=True)

    # Initialization of a global model
    global_model = SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)
    global_threshold = 1.0  # Arbitrary initialization of the threshold of the global model

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt = \
        get_autoencoder_dataloaders(args, all_devices, ctp=ctp, color=Color.YELLOW)
    dataloaders_train = dataloaders_train[:8]  # We only see the data from 8 devices during training, so that the data from the last device is unseen
    dataloaders_opt = dataloaders_opt[:8]
    ctp.print('\n')

    for federation_round in range(args.federation_rounds):
        ctp.print('\t\t\t\t\tFederation round [{}/{}]'.format(federation_round + 1, args.federation_rounds), bold=True)
        ctp.add_bar(Color.BOLD)
        ctp.print()

        # Distribute the global model to all clients
        models = [deepcopy(global_model) for _ in all_devices[:8]]

        # Local training of each client
        multitrain_autoencoders(trains=zip(['with data from ' + device for device in all_devices[:8]], dataloaders_train, models),
                                args=args, ctp=ctp, lr_factor=(args.gamma_round ** federation_round),
                                main_title='Training the different clients', color=Color.GREEN)
        ctp.print('\n')

        # Computation of the thresholds
        thresholds = compute_thresholds(opts=zip(['with opt data from ' + device for device in all_devices[:8]], dataloaders_opt, models),
                                        ctp=ctp, color=Color.RED)
        ctp.print('\n')

        # Experiment 1: test all 8 trained clients on the data of the unseen device (9th)
        multitest_autoencoders(tests=zip(['Model trained on dataset ' + device for device in all_devices],
                                         [dataloaders_benign_test[8] for _ in range(8)],
                                         [dataloaders_mirai[8] for _ in range(8)],
                                         [dataloaders_gafgyt[8] for _ in range(8)],
                                         models, thresholds),
                               ctp=ctp, main_title='Testing different clients on their own data', color=Color.BLUE)
        ctp.print('\n')

        # Experiment 2: test the global model on the data of all devices, before it has been averaged
        multitest_autoencoders(tests=zip(['Data from device ' + device for device in all_devices],
                                         dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt,
                                         [global_model for _ in range(9)],
                                         [global_threshold for _ in range(9)]),
                               ctp=ctp, main_title='Testing global model (before averaging) on data from different devices', color=Color.DARKCYAN)
        ctp.print('\n')

        # Federated averaging
        federated_averaging(global_model, models)
        global_threshold = sum(thresholds) / len(thresholds)

        # Experiment 3: test the global model on the data of all devices, after it has been averaged
        multitest_autoencoders(tests=zip(['Data from device ' + device for device in all_devices],
                                         dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt,
                                         [global_model for _ in range(9)],
                                         [global_threshold for _ in range(9)]),
                               ctp=ctp, main_title='Testing global model (after averaging) on data from different devices', color=Color.CYAN)
        ctp.remove_header()
        ctp.print('\n')
