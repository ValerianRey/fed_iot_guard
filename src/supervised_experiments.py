from copy import deepcopy
from types import SimpleNamespace
from typing import Tuple, List

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from architectures import BinaryClassifier, NormalizingModel
from data import ClientData, FederationData, device_names
from federated_util import federated_averaging
from metrics import BinaryClassificationResult
from ml import set_model_sub_div, set_models_sub_divs
from print_util import print_federation_round, print_rates
from supervised_data import get_train_test_dls, get_train_dl, get_test_dl
from supervised_ml import multitrain_classifiers, multitest_classifiers, train_classifier, test_classifier


def local_classifier_train_val(train_data: ClientData, val_data: ClientData, params: SimpleNamespace) -> BinaryClassificationResult:
    # Creating the dataloaders
    train_dl = get_train_dl(train_data, params.train_bs, params.cuda)
    val_dl = get_test_dl(val_data, params.test_bs, params.cuda)

    # Initialize the model and compute the normalization values with the client's local training data
    model = NormalizingModel(BinaryClassifier(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                             sub=torch.zeros(params.n_features), div=torch.ones(params.n_features))

    if params.cuda:
        model = model.cuda()

    set_model_sub_div(params.normalization, model, train_dl)

    # Local training
    Ctp.enter_section('Training for {} epochs'.format(params.epochs), color=Color.GREEN)
    train_classifier(model, params, train_dl)
    Ctp.exit_section()

    # Local validation
    Ctp.print('Validating')
    result = test_classifier(model, val_dl)
    print_rates(result)

    return result


def local_classifiers_train_test(train_data: FederationData, local_test_data: FederationData,
                                 new_test_data: ClientData, params: SimpleNamespace) \
        -> Tuple[BinaryClassificationResult, BinaryClassificationResult]:
    # Creating the dataloaders
    train_dls, local_test_dls = get_train_test_dls(train_data, local_test_data, params.train_bs, params.test_bs, params.cuda)
    new_test_dl = get_test_dl(new_test_data, params.test_bs, params.cuda)

    # Initialize the models and compute the normalization values with each client's local training data
    n_clients = len(params.clients_devices)
    models = [NormalizingModel(BinaryClassifier(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                               sub=torch.zeros(params.n_features), div=torch.ones(params.n_features)) for _ in range(n_clients)]

    if params.cuda:
        models = [model.cuda() for model in models]

    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    # Training
    multitrain_classifiers(trains=list(zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                            for i, client_devices in enumerate(params.clients_devices)],
                                           train_dls, models)),
                           params=params, main_title='Training the clients', color=Color.GREEN)

    # Local testing
    local_result = multitest_classifiers(tests=list(zip(['Testing client {} on: '.format(i + 1) + device_names(client_devices)
                                                         for i, client_devices in enumerate(params.clients_devices)],
                                                        local_test_dls, models)),
                                         main_title='Testing the clients on their own devices', color=Color.BLUE)

    # New devices testing
    new_devices_result = multitest_classifiers(
        tests=list(zip(['Testing client {} on: '.format(i + 1) + device_names(params.test_devices) for i in range(n_clients)],
                       [new_test_dl for _ in range(n_clients)], models)),
        main_title='Testing the clients on the new devices: ' + device_names(params.test_devices),
        color=Color.DARK_CYAN)

    return local_result, new_devices_result


def federated_classifiers_train_test(train_data: FederationData, local_test_data: FederationData,
                                     new_test_data: ClientData, params: SimpleNamespace) \
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult]]:
    # Creating the dataloaders
    train_dls, local_test_dls = get_train_test_dls(train_data, local_test_data, params.train_bs, params.test_bs, params.cuda)
    new_test_dl = get_test_dl(new_test_data, params.test_bs, params.cuda)

    # Initialization of a global model
    n_clients = len(params.clients_devices)
    global_model = NormalizingModel(BinaryClassifier(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                                    sub=torch.zeros(params.n_features), div=torch.ones(params.n_features))

    if params.cuda:
        global_model = global_model.cuda()

    models = [deepcopy(global_model) for _ in range(n_clients)]
    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    # Initialization of the results
    local_results, new_devices_results = [], []

    for federation_round in range(params.federation_rounds):
        print_federation_round(federation_round, params.federation_rounds)

        # Local training of each client
        multitrain_classifiers(trains=list(zip(['Training client {} on: '.format(i + 1) + device_names(client_devices)
                                                for i, client_devices in enumerate(params.clients_devices)],
                                               train_dls, models)),
                               params=params, lr_factor=(params.gamma_round ** federation_round),
                               main_title='Training the clients', color=Color.GREEN)

        # Federated averaging
        params.aggregation_function(global_model, models)

        # Distribute the global model back to each client
        models = [deepcopy(global_model) for _ in range(n_clients)]

        # Global model testing on each client's data
        result = multitest_classifiers(tests=list(zip(['Testing global model on: ' + device_names(client_devices)
                                                       for client_devices in params.clients_devices],
                                                      local_test_dls, [global_model for _ in range(n_clients)])),
                                       main_title='Testing the global model on data from all clients', color=Color.BLUE)
        local_results.append(result)

        # Global model testing on new devices
        result = multitest_classifiers(
            tests=list(zip(['Testing global model on: ' + device_names(params.test_devices)], [new_test_dl], [global_model])),
            main_title='Testing the global model on the new devices: ' + device_names(params.test_devices),
            color=Color.DARK_CYAN)
        new_devices_results.append(result)

        Ctp.exit_section()

    return local_results, new_devices_results
