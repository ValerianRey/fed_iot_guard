from copy import deepcopy
from types import SimpleNamespace
from typing import Tuple, List

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from src.anomaly_detection_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds, train_autoencoder, \
    compute_reconstruction_losses
from src.architectures import SimpleAutoencoder, NormalizingModel
from src.data import device_names, split_clients_data, ClientData, FederationData
from src.federated_util import federated_averaging
from src.general_ml import set_models_sub_divs, set_model_sub_div
from src.metrics import BinaryClassificationResult
from src.print_util import print_federation_round
from src.unsupervised_data import get_train_dl, get_val_dl, get_test_dls_dict, get_train_val_test_dls


def local_autoencoder_train_val(train_data: ClientData, val_data: ClientData, args: SimpleNamespace) -> float:
    # Create the dataloaders
    train_dl = get_train_dl(train_data, args.train_bs)
    val_dl = get_val_dl(val_data, args.test_bs)

    # Initialize the model and compute the normalization values with the client's local training data
    model = NormalizingModel(SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers),
                             sub=torch.zeros(args.n_features), div=torch.ones(args.n_features))
    set_model_sub_div(args, model, train_dl)

    # Local training
    Ctp.print('Training')
    train_autoencoder(model, args, train_dl)

    # Local validation
    Ctp.print('Validating')
    losses = compute_reconstruction_losses(model, val_dl)
    loss = (sum(losses) / len(losses)).item()
    Ctp.print("Loss: {:.5f}".format(loss))

    return loss


def local_autoencoders_train_test(train_val_data: FederationData, local_test_data: FederationData, new_test_data: ClientData, args: SimpleNamespace) \
        -> Tuple[BinaryClassificationResult, BinaryClassificationResult]:
    # Split train data between actual train and opt
    # TODO: try to place p_test and p_unused in the args or in the parameters
    #  (ideally they should be the same as those used during gs, but maybe not necessarily)
    train_data, val_data = split_clients_data(train_val_data, p_test=0.5, p_unused=0.0)

    # Creating the dataloaders
    train_dls, val_dls, local_test_dls_dicts = get_train_val_test_dls(train_data, val_data, local_test_data, args.train_bs, args.test_bs)
    new_test_dl_dicts = get_test_dls_dict(new_test_data, args.test_bs)

    # Initialize the models and compute the normalization values with each client's local training data
    n_clients = len(args.clients_devices)
    models = [NormalizingModel(SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers),
                               sub=torch.zeros(args.n_features), div=torch.ones(args.n_features)) for _ in range(n_clients)]
    set_models_sub_divs(args, models, train_dls, color=Color.RED)

    # Local training of the autoencoder
    multitrain_autoencoders(trains=list(zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                             for i, client_devices in enumerate(args.clients_devices)], train_dls, models)),
                            args=args, main_title='Training the clients', color=Color.GREEN)

    # Computation of the thresholds
    thresholds = compute_thresholds(opts=list(zip(['Computing threshold for client {} on: '.format(i + 1) + device_names(client_devices)
                                                   for i, client_devices in enumerate(args.clients_devices)], val_dls, models)),
                                    main_title='Computing the thresholds', color=Color.DARK_PURPLE)

    # Local testing of each autoencoder
    local_result = multitest_autoencoders(tests=list(zip(['Testing client {} on: '.format(i + 1) + device_names(client_devices)
                                                          for i, client_devices in enumerate(args.clients_devices)],
                                                         local_test_dls_dicts, models, thresholds)),
                                          main_title='Testing the clients on their own devices', color=Color.BLUE)

    # New devices testing
    new_devices_result = multitest_autoencoders(
        tests=list(zip(['Testing client {} on: '.format(i + 1) + device_names(args.test_devices) for i in range(n_clients)],
                       [new_test_dl_dicts for _ in range(n_clients)], models, thresholds)),
        main_title='Testing the clients on the new devices: ' + device_names(args.test_devices), color=Color.DARK_CYAN)

    return local_result, new_devices_result


def federated_autoencoders_train_test(train_val_data: FederationData, local_test_data: FederationData, new_test_data: ClientData,
                                      args: SimpleNamespace) -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult]]:
    # Split train data between actual train and opt
    # TODO: try to place p_test and p_unused in the args or in the parameters
    #  (ideally they should be the same as those used during gs, but maybe not necessarily)
    train_val_data, val_data = split_clients_data(train_val_data, p_test=0.5, p_unused=0.0)

    # Creating the dataloaders
    train_dls, val_dls, local_test_dls_dicts = get_train_val_test_dls(train_val_data, val_data, local_test_data, args.train_bs, args.test_bs)
    new_test_dls_dict = get_test_dls_dict(new_test_data, args.test_bs)

    # Initialization of a global model
    n_clients = len(args.clients_devices)
    global_model = NormalizingModel(SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers),
                                    sub=torch.zeros(args.n_features), div=torch.ones(args.n_features))
    models = [deepcopy(global_model) for _ in range(n_clients)]
    set_models_sub_divs(args, models, train_dls, color=Color.RED)

    # Initialization of the results
    local_results, new_devices_results = [], []

    for federation_round in range(args.federation_rounds):
        print_federation_round(federation_round, args.federation_rounds)

        # Local training of each client
        multitrain_autoencoders(trains=list(zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                                 for i, client_devices in enumerate(args.clients_devices)],
                                                train_dls, models)),
                                args=args, lr_factor=(args.gamma_round ** federation_round),
                                main_title='Training the clients', color=Color.GREEN)

        # Computation of the thresholds
        thresholds = compute_thresholds(opts=list(zip(['Computing threshold for client {} on: '.format(i + 1) + device_names(client_devices)
                                                       for i, client_devices in enumerate(args.clients_devices)], val_dls, models)),
                                        main_title='Computing the thresholds', color=Color.DARK_PURPLE)

        # Federated averaging
        federated_averaging(global_model, models)
        global_threshold = sum(thresholds) / len(thresholds)

        # Distribute the global model back to each client
        models = [deepcopy(global_model) for _ in range(n_clients)]

        # Global model testing on each client's data
        local_results.append(multitest_autoencoders(tests=list(zip(['Testing global model on: ' + device_names(client_devices)
                                                                    for client_devices in args.clients_devices],
                                                                   local_test_dls_dicts, [global_model for _ in range(n_clients)],
                                                                   [global_threshold for _ in range(n_clients)])),
                                                    main_title='Testing the global model on data from all clients', color=Color.BLUE))

        # Global model testing on new devices
        new_devices_results.append(multitest_autoencoders(tests=list(zip(['Testing global model on: ' + device_names(args.test_devices)],
                                                                         [new_test_dls_dict], [global_model], [global_threshold])),
                                                          main_title='Testing the global model on the new devices: ' + device_names(
                                                              args.test_devices),
                                                          color=Color.DARK_CYAN))
        Ctp.exit_section()

    return local_results, new_devices_results
