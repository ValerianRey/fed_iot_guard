from types import SimpleNamespace
from typing import Tuple, List

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from architectures import BinaryClassifier, NormalizingModel
from data import ClientData, FederationData, device_names, get_benign_attack_samples_per_device
from federated_util import init_federated_models, model_aggregation, select_mimicked_client, model_poisoning
from metrics import BinaryClassificationResult
from ml import set_model_sub_div, set_models_sub_divs
from print_util import print_federation_round, print_rates, print_federation_epoch
from supervised_data import get_train_dl, get_test_dl, prepare_dataloaders
from supervised_ml import multitrain_classifiers, multitest_classifiers, train_classifier, test_classifier, train_classifiers_fedsgd


def local_classifier_train_val(train_data: ClientData, val_data: ClientData, params: SimpleNamespace) -> BinaryClassificationResult:
    p_train = params.p_train_val * (1. - params.val_part)
    p_val = params.p_train_val * params.val_part

    # Creating the dataloaders
    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=p_train,
                                                                                                benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    train_dl = get_train_dl(train_data, params.train_bs, benign_samples_per_device=benign_samples_per_device,
                            attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=p_val, benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    val_dl = get_test_dl(val_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                         attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    # Initialize the model and compute the normalization values with the client's local training data
    model = NormalizingModel(BinaryClassifier(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                             sub=torch.zeros(params.n_features), div=torch.ones(params.n_features))

    if params.cuda:
        model = model.cuda()

    set_model_sub_div(params.normalization, model, train_dl)

    # Local training
    Ctp.enter_section('Training for {} epochs with {} samples'.format(params.epochs, len(train_dl.dataset[:][0])), color=Color.GREEN)
    train_classifier(model, params, train_dl)
    Ctp.exit_section()

    # Local validation
    Ctp.print('Validating with {} samples'.format(len(val_dl.dataset[:][0])))
    result = test_classifier(model, val_dl)
    print_rates(result)

    return result


def local_classifiers_train_test(train_data: FederationData, local_test_data: FederationData,
                                 new_test_data: ClientData, params: SimpleNamespace) \
        -> Tuple[BinaryClassificationResult, BinaryClassificationResult]:
    train_dls, local_test_dls, new_test_dl = prepare_dataloaders(train_data, local_test_data, new_test_data, params, federated=False)

    # Initialize the models and compute the normalization values with each client's local training data
    n_clients = len(params.clients_devices)
    models = [NormalizingModel(BinaryClassifier(activation_function=params.activation_fn, hidden_layers=params.hidden_layers),
                               sub=torch.zeros(params.n_features), div=torch.ones(params.n_features)) for _ in range(n_clients)]

    if params.cuda:
        models = [model.cuda() for model in models]

    set_models_sub_divs(params.normalization, models, train_dls, color=Color.RED)

    # Training
    multitrain_classifiers(trains=list(zip(['Training client {} on: '.format(i) + device_names(client_devices)
                                            for i, client_devices in enumerate(params.clients_devices)],
                                           train_dls, models)),
                           params=params, main_title='Training the clients', color=Color.GREEN)

    # Local testing
    local_result = multitest_classifiers(tests=list(zip(['Testing client {} on: '.format(i) + device_names(client_devices)
                                                         for i, client_devices in enumerate(params.clients_devices)],
                                                        local_test_dls, models)),
                                         main_title='Testing the clients on their own devices', color=Color.BLUE)

    # New devices testing
    new_devices_result = multitest_classifiers(
        tests=list(zip(['Testing client {} on: '.format(i) + device_names(params.test_devices) for i in range(n_clients)],
                       [new_test_dl for _ in range(n_clients)], models)),
        main_title='Testing the clients on the new devices: ' + device_names(params.test_devices),
        color=Color.DARK_CYAN)

    return local_result, new_devices_result


def federated_testing(global_model: torch.nn.Module, local_test_dls: List[DataLoader], new_test_dl: DataLoader,
                      params: SimpleNamespace, local_results: List[BinaryClassificationResult],
                      new_devices_results: List[BinaryClassificationResult]) -> None:

    # Global model testing on each client's data
    tests = []
    for client_id, client_devices in enumerate(params.clients_devices):
        if client_id not in params.malicious_clients:
            tests.append(('Testing global model on: ' + device_names(client_devices), local_test_dls[client_id], global_model))

    result = multitest_classifiers(tests=tests,
                                   main_title='Testing the global model on data from all clients', color=Color.BLUE)
    local_results.append(result)

    # Global model testing on new devices
    result = multitest_classifiers(
        tests=list(zip(['Testing global model on: ' + device_names(params.test_devices)], [new_test_dl], [global_model])),
        main_title='Testing the global model on the new devices: ' + device_names(params.test_devices),
        color=Color.DARK_CYAN)
    new_devices_results.append(result)


def fedavg_classifiers_train_test(train_data: FederationData, local_test_data: FederationData,
                                  new_test_data: ClientData, params: SimpleNamespace) \
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult]]:
    # Preparation of the dataloaders
    train_dls, local_test_dls, new_test_dl = prepare_dataloaders(train_data, local_test_data, new_test_data, params, federated=True)

    # Initialization of the models
    global_model, models = init_federated_models(train_dls, params, architecture=BinaryClassifier)

    # Initialization of the results
    local_results, new_devices_results = [], []

    # Selection of a client to mimic in case we use the mimic attack
    mimicked_client_id = select_mimicked_client(params)

    for federation_round in range(params.federation_rounds):
        print_federation_round(federation_round, params.federation_rounds)

        # Local training of each client
        multitrain_classifiers(trains=list(zip(['Training client {} on: '.format(i) + device_names(client_devices)
                                                for i, client_devices in enumerate(params.clients_devices)],
                                               train_dls, models)),
                               params=params, lr_factor=(params.gamma_round ** federation_round),
                               main_title='Training the clients', color=Color.GREEN)

        # Model poisoning attacks
        models = model_poisoning(global_model, models, params, mimicked_client_id=mimicked_client_id, verbose=True)

        # Aggregation
        global_model, models = model_aggregation(global_model, models, params, verbose=True)

        # Testing
        federated_testing(global_model, local_test_dls, new_test_dl, params, local_results, new_devices_results)

        Ctp.exit_section()

    return local_results, new_devices_results


def fedsgd_classifiers_train_test(train_data: FederationData, local_test_data: FederationData,
                                  new_test_data: ClientData, params: SimpleNamespace) \
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult]]:
    # Preparation of the dataloaders
    train_dls, local_test_dls, new_test_dl = prepare_dataloaders(train_data, local_test_data, new_test_data, params, federated=True)

    # Initialization of the models
    global_model, models = init_federated_models(train_dls, params, architecture=BinaryClassifier)

    # Initialization of the results
    local_results, new_devices_results = [], []

    # Selection of a client to mimic in case we use the mimic attack
    mimicked_client_id = select_mimicked_client(params)

    for epoch in range(params.epochs):
        print_federation_epoch(epoch, params.epochs)
        lr_factor = params.lr_scheduler_params['gamma'] ** (epoch // params.lr_scheduler_params['step_size'])
        global_model, models = train_classifiers_fedsgd(global_model, models, train_dls, params, epoch,
                                                        lr_factor=lr_factor, mimicked_client_id=mimicked_client_id)
        federated_testing(global_model, local_test_dls, new_test_dl, params, local_results, new_devices_results)
        Ctp.exit_section()

    return local_results, new_devices_results
