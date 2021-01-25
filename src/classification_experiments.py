from copy import deepcopy

from architectures import BinaryClassifier
from classification_ml import multitrain_classifiers, multitest_classifiers
from data import all_devices, get_classifier_dataloaders
from federated_util import federated_averaging
from print_util import Color, ContextPrinter


def device_names(device_ids):
    return ', '.join([all_devices[device_id] for device_id in device_ids])


def local_classifiers(args):
    ctp = ContextPrinter()
    n_clients = len(args.clients_devices)
    if n_clients == 1:
        ctp.print('\n\t\t\t\t\tSINGLE CLASSIFIER\n', bold=True)
    else:
        ctp.print('\n\t\t\t\t\tMULTIPLE CLASSIFIERS\n', bold=True)

    # Initialize the model
    models = [BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers) for _ in range(n_clients)]

    # Loading the data and creating the dataloaders
    clients_dataloaders_train, clients_dataloaders_test, new_dataloader_test = get_classifier_dataloaders(args, ctp=ctp, color=Color.YELLOW)
    ctp.print('\n')

    # Training
    multitrain_classifiers(trains=zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                       for i, client_devices in enumerate(args.clients_devices)],
                                      clients_dataloaders_train, models),
                           args=args, ctp=ctp, main_title='Training the clients', color=Color.GREEN)
    ctp.print('\n')

    # Local testing
    multitest_classifiers(tests=zip(['Testing client {} on: '.format(i) + device_names(client_devices)
                                     for i, client_devices in enumerate(args.clients_devices)],
                                    clients_dataloaders_test, models),
                          ctp=ctp, main_title='Testing the clients on their own devices', color=Color.BLUE)
    ctp.print('\n')

    # New devices testing
    multitest_classifiers(tests=zip(['Testing client trained on: ' + device_names(client_devices) for client_devices in args.clients_devices],
                                    [new_dataloader_test for _ in range(n_clients)], models),
                          ctp=ctp, main_title='Testing the clients on the new devices: ' + device_names(args.test_devices), color=Color.DARKCYAN)


def federated_classifiers(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tFEDERATED CLASSIFIERS\n', bold=True)
    n_clients = len(args.clients_devices)

    # Initialization of a global model
    global_model = BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    clients_dataloaders_train, clients_dataloaders_test, new_dataloader_test = get_classifier_dataloaders(args, ctp=ctp, color=Color.YELLOW)

    ctp.print('\n')

    for federation_round in range(args.federation_rounds):
        ctp.print('\t\t\t\t\tFederation round [{}/{}]'.format(federation_round + 1, args.federation_rounds), bold=True)
        ctp.add_bar(Color.BOLD)
        ctp.print()

        # Distribute the global model to all clients
        models = [deepcopy(global_model) for _ in range(n_clients)]

        # Local training of each client
        multitrain_classifiers(trains=zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                           for i, client_devices in enumerate(args.clients_devices)],
                                          clients_dataloaders_train, models),
                               args=args, ctp=ctp, lr_factor=(args.gamma_round ** federation_round),
                               main_title='Training the clients', color=Color.GREEN)
        ctp.print('\n')

        # Local testing before federated averaging
        multitest_classifiers(tests=zip(['Testing client {} on: '.format(i) + device_names(client_devices)
                                         for i, client_devices in enumerate(args.clients_devices)],
                                        clients_dataloaders_test, models),
                              ctp=ctp, main_title='Testing the clients on their own devices', color=Color.BLUE)
        ctp.print('\n')

        # New devices testing before federated aggregation
        multitest_classifiers(tests=zip(['Testing client {} on: '.format(i) + device_names(args.test_devices)
                                         for i in range(n_clients)],
                                        [new_dataloader_test for _ in range(len(models))], models),
                              ctp=ctp, main_title='Testing the clients on the new devices: ' + device_names(args.test_devices),
                              color=Color.DARKCYAN)
        ctp.print('\n')
        # Federated averaging
        federated_averaging(global_model, models)

        # Global model testing on each client's data
        multitest_classifiers(tests=zip(['Testing global model on: ' + device_names(client_devices)
                                         for client_devices in args.clients_devices],
                                        clients_dataloaders_test, [global_model for _ in range(n_clients)]),
                              ctp=ctp, main_title='Testing the global model on data from all clients', color=Color.PURPLE)
        ctp.print('\n')

        # Global model testing on new devices
        multitest_classifiers(tests=zip(['Testing global model on: ' + device_names(args.test_devices)], [new_dataloader_test], [global_model]),
                              ctp=ctp, main_title='Testing the global model on the new devices: ' + device_names(args.test_devices),
                              color=Color.CYAN)
        ctp.remove_header()
        ctp.print('\n')
