from copy import deepcopy
from types import SimpleNamespace
from typing import Tuple, List

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from architectures import SimpleAutoencoder, NormalizingModel, Threshold
from data import device_names, split_clients_data, ClientData, FederationData, get_benign_attack_samples_per_device
from metrics import BinaryClassificationResult
from ml import set_models_sub_divs, set_model_sub_div
from print_util import print_federation_round
from unsupervised_data import get_train_dl, get_val_dl, get_test_dls_dict, get_train_dls, \
    get_val_dls, get_test_dls_dicts
from unsupervised_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds, train_autoencoder, \
    compute_reconstruction_losses


def local_autoencoder_train_val(train_data: ClientData, val_data: ClientData, params: SimpleNamespace) -> float:
    p_train = params.p_train_val * (1. - params.val_part)
    p_val = params.p_train_val * params.val_part

    # Create the dataloaders
    benign_samples_per_device, _ = get_benign_attack_samples_per_device(p_split=p_train, benign_prop=1.,
                                                                        samples_per_device=params.samples_per_device)
    train_dl = get_train_dl(train_data, params.train_bs, benign_samples_per_device=benign_samples_per_device, cuda=params.cuda)

    benign_samples_per_device, _ = get_benign_attack_samples_per_device(p_split=p_val, benign_prop=1.,
                                                                        samples_per_device=params.samples_per_device)
    val_dl = get_val_dl(val_data, params.test_bs, benign_samples_per_device=benign_samples_per_device, cuda=params.cuda)

    # Initialize the model and compute the normalization values with the client's local training data
    model = NormalizingModel(SimpleAutoencoder(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                             sub=torch.zeros(params.n_features), div=torch.ones(params.n_features))
    if params.cuda:
        model = model.cuda()

    set_model_sub_div(params.normalization, model, train_dl)

    # Local training
    Ctp.enter_section('Training for {} epochs with {} samples'.format(params.epochs, len(train_dl.dataset[:][0])), color=Color.GREEN)
    train_autoencoder(model, params, train_dl)
    Ctp.exit_section()

    # Local validation
    Ctp.print("Validating with {} samples".format(len(val_dl.dataset[:][0])))
    losses = compute_reconstruction_losses(model, val_dl)
    loss = (sum(losses) / len(losses)).item()
    Ctp.print("Validation loss: {:.5f}".format(loss))

    return loss


def prepare_dataloaders(train_val_data: FederationData, local_test_data: FederationData, new_test_data: ClientData, params: SimpleNamespace):
    # Split train data between actual train and the set that will be used to search the threshold
    train_data, threshold_data = split_clients_data(train_val_data, p_second_split=params.threshold_part, p_unused=0.0)

    p_train = params.p_train_val * (1. - params.threshold_part)
    p_threshold = params.p_train_val * params.threshold_part

    # Creating the dataloaders
    benign_samples_per_device, _ = get_benign_attack_samples_per_device(p_split=p_train,
                                                                        benign_prop=1., samples_per_device=params.samples_per_device)
    train_dls = get_train_dls(train_data, params.train_bs, benign_samples_per_device=benign_samples_per_device, cuda=params.cuda)

    benign_samples_per_device, _ = get_benign_attack_samples_per_device(p_split=p_threshold,
                                                                        benign_prop=1., samples_per_device=params.samples_per_device)
    threshold_dls = get_val_dls(threshold_data, params.test_bs, benign_samples_per_device=benign_samples_per_device, cuda=params.cuda)

    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_test, benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    local_test_dls_dicts = get_test_dls_dicts(local_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                                              attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    new_test_dls_dict = get_test_dls_dict(new_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                                          attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    return train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict


def local_autoencoders_train_test(train_val_data: FederationData, local_test_data: FederationData, new_test_data: ClientData,
                                  params: SimpleNamespace) -> Tuple[BinaryClassificationResult, BinaryClassificationResult, List[float]]:
    # Prepare the dataloaders
    train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict = prepare_dataloaders(train_val_data, local_test_data, new_test_data, params)

    # Initialize the models and compute the normalization values with each client's local training data
    n_clients = len(params.clients_devices)
    models = [NormalizingModel(SimpleAutoencoder(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                               sub=torch.zeros(params.n_features), div=torch.ones(params.n_features)) for _ in range(n_clients)]

    if params.cuda:
        models = [model.cuda() for model in models]

    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    # Local training of the autoencoder
    multitrain_autoencoders(trains=list(zip(['Training client {} on: '.format(i) + device_names(client_devices)
                                             for i, client_devices in enumerate(params.clients_devices)], train_dls, models)),
                            params=params, main_title='Training the clients', color=Color.GREEN)

    # Computation of the thresholds
    thresholds = compute_thresholds(opts=list(zip(['Computing threshold for client {} on: '.format(i) + device_names(client_devices)
                                                   for i, client_devices in enumerate(params.clients_devices)], threshold_dls, models)),
                                    quantile=params.quantile,
                                    main_title='Computing the thresholds', color=Color.DARK_PURPLE)

    # Local testing of each autoencoder
    local_result = multitest_autoencoders(tests=list(zip(['Testing client {} on: '.format(i) + device_names(client_devices)
                                                          for i, client_devices in enumerate(params.clients_devices)],
                                                         local_test_dls_dicts, models, thresholds)),
                                          main_title='Testing the clients on their own devices', color=Color.BLUE)

    # New devices testing
    new_devices_result = multitest_autoencoders(
        tests=list(zip(['Testing client {} on: '.format(i) + device_names(params.test_devices) for i in range(n_clients)],
                       [new_test_dls_dict for _ in range(n_clients)], models, thresholds)),
        main_title='Testing the clients on the new devices: ' + device_names(params.test_devices), color=Color.DARK_CYAN)

    return local_result, new_devices_result, [threshold.threshold.item() for threshold in thresholds]


def federated_autoencoders_train_test(train_val_data: FederationData, local_test_data: FederationData,
                                      new_test_data: ClientData, params: SimpleNamespace)\
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult], List[float]]:
    # Prepare the dataloaders
    train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict = prepare_dataloaders(train_val_data, local_test_data, new_test_data, params)

    # Initialization of a global model
    n_clients = len(params.clients_devices)
    global_model = NormalizingModel(SimpleAutoencoder(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                                    sub=torch.zeros(params.n_features), div=torch.ones(params.n_features))
    global_threshold = Threshold(torch.tensor(0.))

    if params.cuda:
        global_model = global_model.cuda()

    models = [deepcopy(global_model) for _ in range(n_clients)]
    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    # Initialization of the results
    local_results, new_devices_results, global_thresholds = [], [], []

    for federation_round in range(params.federation_rounds):
        print_federation_round(federation_round, params.federation_rounds)

        # Local training of each client
        multitrain_autoencoders(trains=list(zip(['Training client {} on: '.format(i) + device_names(client_devices)
                                                 for i, client_devices in enumerate(params.clients_devices)],
                                                train_dls, models)),
                                params=params, lr_factor=(params.gamma_round ** federation_round),
                                main_title='Training the clients', color=Color.GREEN)

        # Federated aggregation of the models
        params.aggregation_function(global_model, models)

        # Distribute the global model back to each client
        models = [deepcopy(global_model) for _ in range(n_clients)]

        # Computation of the thresholds
        thresholds = compute_thresholds(opts=list(zip(['Computing threshold for client {} on: '.format(i) + device_names(client_devices)
                                                       for i, client_devices in enumerate(params.clients_devices)], threshold_dls, models)),
                                        quantile=params.quantile,
                                        main_title='Computing the thresholds', color=Color.DARK_PURPLE)

        # Federated aggregation of the thresholds
        params.aggregation_function(global_threshold, thresholds)
        Ctp.print('Global threshold: {:.6f}'.format(global_threshold.threshold.item()))
        global_thresholds.append(global_threshold.threshold.item())
        # There is no need to distribute the global threshold back to each client since they will compute it again from scratch at the next iteration
        # But in reality it's like if we transmitted them back because the local testing is made with the global threshold

        # Global model testing on each client's data
        local_results.append(multitest_autoencoders(tests=list(zip(['Testing global model on: ' + device_names(client_devices)
                                                                    for client_devices in params.clients_devices],
                                                                   local_test_dls_dicts, [global_model for _ in range(n_clients)],
                                                                   [global_threshold for _ in range(n_clients)])),
                                                    main_title='Testing the global model on data from all clients', color=Color.BLUE))

        # Global model testing on new devices
        new_devices_results.append(multitest_autoencoders(tests=list(zip(['Testing global model on: ' + device_names(params.test_devices)],
                                                                         [new_test_dls_dict], [global_model], [global_threshold])),
                                                          main_title='Testing the global model on the new devices: ' + device_names(
                                                              params.test_devices),
                                                          color=Color.DARK_CYAN))
        Ctp.exit_section()

    return local_results, new_devices_results, global_thresholds
