from copy import deepcopy

import torch

from architectures import BinaryClassifier, NormalizingModel
from classification_ml import multitrain_classifiers, multitest_classifiers
from general_ml import set_models_sub_divs
from data import get_supervised_dataloaders, device_names
from federated_util import federated_averaging
from print_util import Color, ContextPrinter, print_federation_round


def local_classifiers(args):
    ctp = ContextPrinter()
    n_clients = len(args.clients_devices)
    if n_clients == 1:
        ctp.print('\n\t\t\t\t\tSINGLE CLASSIFIER\n', bold=True)
    else:
        ctp.print('\n\t\t\t\t\tMULTIPLE CLASSIFIERS\n', bold=True)

    # Loading the data and creating the dataloaders
    clients_dl_train, clients_dl_test, new_dl_test = get_supervised_dataloaders(args, ctp=ctp, color=Color.YELLOW)
    ctp.print('\n')

    # Initialize the models and compute the normalization values with each client's local training data
    models = [NormalizingModel(BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers),
                               sub=torch.zeros(args.n_features), div=torch.ones(args.n_features)) for _ in range(n_clients)]
    set_models_sub_divs(args, models, clients_dl_train, ctp, color=Color.RED)
    ctp.print('\n')

    # Training
    multitrain_classifiers(trains=zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                       for i, client_devices in enumerate(args.clients_devices)],
                                      clients_dl_train, models),
                           args=args, ctp=ctp, main_title='Training the clients', color=Color.GREEN)
    ctp.print('\n')

    # Local testing
    multitest_classifiers(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(client_devices)
                                     for i, client_devices in enumerate(args.clients_devices)],
                                    clients_dl_test, models),
                          ctp=ctp, main_title='Testing the clients on their own devices', color=Color.BLUE)
    ctp.print('\n')

    # New devices testing
    multitest_classifiers(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(args.test_devices) for i in range(n_clients)],
                                    [new_dl_test for _ in range(n_clients)], models),
                          ctp=ctp, main_title='Testing the clients on the new devices: ' + device_names(args.test_devices), color=Color.DARKCYAN)


def federated_classifiers(args):
    ctp = ContextPrinter()
    n_clients = len(args.clients_devices)
    ctp.print('\n\t\t\t\t\tFEDERATED CLASSIFIERS\n', bold=True)

    # Loading the data and creating the dataloaders
    clients_dl_train, clients_dl_test, new_dl_test = get_supervised_dataloaders(args, ctp=ctp, color=Color.YELLOW)
    ctp.print('\n')

    # Initialization of a global model
    global_model = NormalizingModel(BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers),
                                    sub=torch.zeros(args.n_features), div=torch.ones(args.n_features))
    models = [deepcopy(global_model) for _ in range(n_clients)]
    set_models_sub_divs(args, models, clients_dl_train, ctp, Color.RED)
    ctp.print('\n')

    for federation_round in range(args.federation_rounds):
        print_federation_round(federation_round, args.federation_rounds, ctp)

        # Local training of each client
        multitrain_classifiers(trains=zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                           for i, client_devices in enumerate(args.clients_devices)],
                                          clients_dl_train, models),
                               args=args, ctp=ctp, lr_factor=(args.gamma_round ** federation_round),
                               main_title='Training the clients', color=Color.GREEN)
        ctp.print('\n')

        # Local testing before federated averaging
        multitest_classifiers(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(client_devices)
                                         for i, client_devices in enumerate(args.clients_devices)],
                                        clients_dl_test, models),
                              ctp=ctp, main_title='Testing the clients on their own devices', color=Color.BLUE)
        ctp.print('\n')

        # New devices testing before federated aggregation
        multitest_classifiers(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(args.test_devices)
                                         for i in range(n_clients)],
                                        [new_dl_test for _ in range(len(models))], models),
                              ctp=ctp, main_title='Testing the clients on the new devices: ' + device_names(args.test_devices),
                              color=Color.DARKCYAN)
        ctp.print('\n')

        # Federated averaging
        federated_averaging(global_model, models)

        # Distribute the global model back to each client
        models = [deepcopy(global_model) for _ in range(n_clients)]

        # Global model testing on each client's data
        multitest_classifiers(tests=zip(['Testing global model on: ' + device_names(client_devices)
                                         for client_devices in args.clients_devices],
                                        clients_dl_test, [global_model for _ in range(n_clients)]),
                              ctp=ctp, main_title='Testing the global model on data from all clients', color=Color.PURPLE)
        ctp.print('\n')

        # Global model testing on new devices
        multitest_classifiers(tests=zip(['Testing global model on: ' + device_names(args.test_devices)], [new_dl_test], [global_model]),
                              ctp=ctp, main_title='Testing the global model on the new devices: ' + device_names(args.test_devices),
                              color=Color.CYAN)
        ctp.remove_header()
        ctp.print('\n')
