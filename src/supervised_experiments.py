from copy import deepcopy
from types import SimpleNamespace
from typing import Tuple, List

import numpy as np
import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from architectures import BinaryClassifier, NormalizingModel
from data import ClientData, FederationData, device_names, get_benign_attack_samples_per_device
from federated_util import model_update_scaling, model_canceling_attack, s_resampling, mimic_attack
from metrics import BinaryClassificationResult
from ml import set_model_sub_div, set_models_sub_divs
from print_util import print_federation_round, print_rates
from supervised_data import get_train_dl, get_test_dl, get_train_dls, get_test_dls
from supervised_ml import multitrain_classifiers, multitest_classifiers, train_classifier, test_classifier


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
    # Creating the dataloaders
    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_train_val,
                                                                                                benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    train_dls = get_train_dls(train_data, params.train_bs, malicious_clients=set(), benign_samples_per_device=benign_samples_per_device,
                              attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_test, benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    local_test_dls = get_test_dls(local_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                                  attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    new_test_dl = get_test_dl(new_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                              attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

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


def federated_classifiers_train_test(train_data: FederationData, local_test_data: FederationData,
                                     new_test_data: ClientData, params: SimpleNamespace) \
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult]]:
    # Creating the dataloaders
    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_train_val,
                                                                                                benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    train_dls = get_train_dls(train_data, params.train_bs, benign_samples_per_device=benign_samples_per_device,
                              attack_samples_per_device=attack_samples_per_device, malicious_clients=params.malicious_clients, cuda=params.cuda,
                              poisoning=params.data_poisoning, p_poison=params.p_poison)

    benign_samples_per_device, attack_samples_per_device = get_benign_attack_samples_per_device(p_split=params.p_test,
                                                                                                benign_prop=params.benign_prop,
                                                                                                samples_per_device=params.samples_per_device)
    local_test_dls = get_test_dls(local_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                                  attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

    new_test_dl = get_test_dl(new_test_data, params.test_bs, benign_samples_per_device=benign_samples_per_device,
                              attack_samples_per_device=attack_samples_per_device, cuda=params.cuda)

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

    # Selection of a client to mimic in case we use the mimic attack
    honest_client_ids = [client_id for client_id in range(n_clients) if client_id not in params.malicious_clients]
    mimicked_client_id = np.random.choice(honest_client_ids)

    for federation_round in range(params.federation_rounds):
        print_federation_round(federation_round, params.federation_rounds)

        # Local training of each client
        multitrain_classifiers(trains=list(zip(['Training client {} on: '.format(i) + device_names(client_devices)
                                                for i, client_devices in enumerate(params.clients_devices)],
                                               train_dls, models)),
                               params=params, lr_factor=(params.gamma_round ** federation_round),
                               main_title='Training the clients', color=Color.GREEN)

        malicious_clients_models = [model for client_id, model in enumerate(models) if client_id in params.malicious_clients]
        n_honest = len(models) - len(malicious_clients_models)

        # Model poisoning attacks
        if params.model_poisoning is not None:
            if params.model_poisoning == 'cancel_attack':
                Ctp.print('Launching cancel attack')
                model_canceling_attack(global_model=global_model, malicious_clients_models=malicious_clients_models, n_honest=n_honest)
            elif params.model_poisoning == 'mimic_attack':
                Ctp.print('Launching mimic attack on client {}'.format(mimicked_client_id))
                mimic_attack(models, params.malicious_clients, mimicked_client_id)
            else:
                raise ValueError('Wrong value for model_poisoning: ' + str(params.model_poisoning))

        # Rescale the model updates of the malicious clients (if any)
        model_update_scaling(global_model=global_model, malicious_clients_models=malicious_clients_models, factor=params.model_update_factor)

        # Federated averaging
        if params.resampling is not None:
            models, indexes = s_resampling(models, params.resampling)
            Ctp.print(indexes)
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
