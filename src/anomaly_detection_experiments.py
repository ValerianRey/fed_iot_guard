from copy import deepcopy
from typing import Tuple, List, Dict

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from anomaly_detection_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds
from architectures import SimpleAutoencoder, NormalizingModel
from data import device_names, split_data
from federated_util import federated_averaging
from general_ml import set_models_sub_divs
from metrics import BinaryClassificationResults
from print_util import print_federation_round
from unsupervised_data import get_all_unsupervised_dls
import numpy as np
from types import SimpleNamespace


def local_autoencoders(train_opt_data: List[Dict[str, np.array]], test_data: List[Dict[str, np.array]], args: SimpleNamespace) \
        -> Tuple[BinaryClassificationResults, BinaryClassificationResults]:

    # Split train data between actual train and opt
    train_data, opt_data = split_data(train_opt_data, p_test=0.5, p_unused=0.0)

    # Create the dataloaders
    clients_dl_train, clients_dl_opt, clients_dls_test, new_dls_test = get_all_unsupervised_dls(train_data, opt_data, test_data,
                                                                                                args.clients_devices, args.test_devices,
                                                                                                args.train_bs, args.test_bs)

    # Initialize the models and compute the normalization values with each client's local training data
    n_clients = len(args.clients_devices)
    models = [NormalizingModel(SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers),
                               sub=torch.zeros(args.n_features), div=torch.ones(args.n_features)) for _ in range(n_clients)]
    set_models_sub_divs(args, models, clients_dl_train, color=Color.RED)

    # Local training of the autoencoder
    multitrain_autoencoders(trains=zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                        for i, client_devices in enumerate(args.clients_devices)], clients_dl_train, models),
                            args=args, main_title='Training the clients', color=Color.GREEN)

    # Computation of the thresholds
    thresholds = compute_thresholds(opts=zip(['Computing threshold for client {} on: '.format(i + 1) + device_names(client_devices)
                                              for i, client_devices in enumerate(args.clients_devices)], clients_dl_opt, models),
                                    main_title='Computing the thresholds', color=Color.DARK_PURPLE)

    # Local testing of each autoencoder
    local_result = multitest_autoencoders(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(client_devices)
                                                     for i, client_devices in enumerate(args.clients_devices)], clients_dls_test, models, thresholds),
                                          main_title='Testing the clients on their own devices', color=Color.BLUE)

    # New devices testing
    new_devices_result = multitest_autoencoders(
        tests=zip(['Testing client {} on: '.format(i + 1) + device_names(args.test_devices) for i in range(n_clients)],
                  [new_dls_test for _ in range(n_clients)], models, thresholds),
        main_title='Testing the clients on the new devices: ' + device_names(args.test_devices), color=Color.DARK_CYAN)

    return local_result, new_devices_result


def federated_autoencoders(train_opt_data: List[Dict[str, np.array]], test_data: List[Dict[str, np.array]], args: SimpleNamespace) \
        -> Tuple[List[BinaryClassificationResults], List[BinaryClassificationResults]]:

    # Split train data between actual train and opt
    train_data, opt_data = split_data(train_opt_data, p_test=0.5, p_unused=0.0)

    # Create the dataloaders
    clients_dl_train, clients_dl_opt, clients_dls_test, new_dls_test = get_all_unsupervised_dls(train_data, opt_data, test_data,
                                                                                                args.clients_devices, args.test_devices,
                                                                                                args.train_bs, args.test_bs)

    # Initialization of a global model
    n_clients = len(args.clients_devices)
    global_model = NormalizingModel(SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers),
                                    sub=torch.zeros(args.n_features), div=torch.ones(args.n_features))
    models = [deepcopy(global_model) for _ in range(n_clients)]
    set_models_sub_divs(args, models, clients_dl_train, color=Color.RED)

    # Initialization of the results
    # Since there is no way for the aggregator to know the results, it's only allowed to choose the last model
    # but we keep all results to be able to report their evolution with respect to the federation round
    local_results, new_devices_results = [], []

    for federation_round in range(args.federation_rounds):
        print_federation_round(federation_round, args.federation_rounds)

        # Local training of each client
        multitrain_autoencoders(trains=zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                            for i, client_devices in enumerate(args.clients_devices)],
                                           clients_dl_train, models),
                                args=args, lr_factor=(args.gamma_round ** federation_round),
                                main_title='Training the clients', color=Color.GREEN)

        # Computation of the thresholds
        thresholds = compute_thresholds(opts=zip(['Computing threshold for client {} on: '.format(i + 1) + device_names(client_devices)
                                                  for i, client_devices in enumerate(args.clients_devices)], clients_dl_opt, models),
                                        main_title='Computing the thresholds', color=Color.DARK_PURPLE)

        # # Local testing before federated averaging
        # multitest_autoencoders(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(client_devices)
        #                                   for i, client_devices in enumerate(args.clients_devices)],
        #                                  clients_dls_test, models, thresholds),
        #                        main_title='Testing the clients on their own devices', color=Color.BLUE)
        #
        # # New devices testing before federated aggregation
        # multitest_autoencoders(tests=zip(['Testing client {} on: '.format(i + 1) + device_names(args.test_devices)
        #                                   for i in range(n_clients)],
        #                                  [new_dls_test for _ in range(len(models))], models, thresholds),
        #                        main_title='Testing the clients on the new devices: ' + device_names(args.test_devices),
        #                        color=Color.DARK_CYAN)

        # Federated averaging
        federated_averaging(global_model, models)
        global_threshold = sum(thresholds) / len(thresholds)

        # Distribute the global model back to each client
        models = [deepcopy(global_model) for _ in range(n_clients)]

        # Global model testing on each client's data
        local_results.append(multitest_autoencoders(tests=zip(['Testing global model on: ' + device_names(client_devices)
                                                               for client_devices in args.clients_devices],
                                                              clients_dls_test, [global_model for _ in range(n_clients)],
                                                              [global_threshold for _ in range(n_clients)]),
                                                    main_title='Testing the global model on data from all clients', color=Color.PURPLE))

        # Global model testing on new devices
        new_devices_results.append(multitest_autoencoders(tests=zip(['Testing global model on: ' + device_names(args.test_devices)],
                                                                    [new_dls_test], [global_model], [global_threshold]),
                                                          main_title='Testing the global model on the new devices: ' + device_names(
                                                              args.test_devices),
                                                          color=Color.CYAN))
        Ctp.exit_section()

    return local_results, new_devices_results
