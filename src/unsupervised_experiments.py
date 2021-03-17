from types import SimpleNamespace
from typing import Tuple, List, Dict

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from architectures import SimpleAutoencoder, NormalizingModel, Threshold
from data import device_names, ClientData, FederationData, get_benign_attack_samples_per_device
from federated_util import init_federated_models, model_aggregation, select_mimicked_client, model_poisoning
from metrics import BinaryClassificationResult
from ml import set_models_sub_divs, set_model_sub_div
from print_util import print_federation_round, print_federation_epoch
from unsupervised_data import get_train_dl, get_val_dl, prepare_dataloaders
from unsupervised_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds, train_autoencoder, \
    compute_reconstruction_losses, train_autoencoders_fedsgd


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


def federated_thresholds(models: List[torch.nn.Module], threshold_dls: List[DataLoader], global_threshold: torch.nn.Module,
                         params: SimpleNamespace, global_thresholds: List[float]) -> None:
    # Computation of the thresholds
    thresholds = compute_thresholds(opts=list(zip(['Computing threshold for client {} on: '.format(i) + device_names(client_devices)
                                                   for i, client_devices in enumerate(params.clients_devices)], threshold_dls, models)),
                                    quantile=params.quantile,
                                    main_title='Computing the thresholds', color=Color.DARK_PURPLE)

    # Aggregation of the thresholds
    global_threshold, thresholds = model_aggregation(global_threshold, thresholds, params, verbose=True)
    Ctp.print('Global threshold: {:.6f}'.format(global_threshold.threshold.item()))
    global_thresholds.append(global_threshold.threshold.item())


def federated_testing(global_model: torch.nn.Module, global_threshold: torch.nn.Module,
                      local_test_dls_dicts: List[Dict[str, DataLoader]], new_test_dls_dict: Dict[str, DataLoader],
                      params: SimpleNamespace, local_results: List[BinaryClassificationResult],
                      new_devices_results: List[BinaryClassificationResult]) -> None:

    # Global model testing on each client's data
    tests = []
    for client_id, client_devices in enumerate(params.clients_devices):
        if client_id not in params.malicious_clients:
            tests.append(('Testing global model on: ' + device_names(client_devices), local_test_dls_dicts[client_id], global_model,
                          global_threshold))

    local_results.append(multitest_autoencoders(tests=tests,
                                                main_title='Testing the global model on data from all clients', color=Color.BLUE))

    # Global model testing on new devices
    new_devices_results.append(multitest_autoencoders(tests=list(zip(['Testing global model on: ' + device_names(params.test_devices)],
                                                                     [new_test_dls_dict], [global_model], [global_threshold])),
                                                      main_title='Testing the global model on the new devices: ' + device_names(
                                                          params.test_devices),
                                                      color=Color.DARK_CYAN))


def fedavg_autoencoders_train_test(train_val_data: FederationData, local_test_data: FederationData,
                                   new_test_data: ClientData, params: SimpleNamespace)\
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult], List[float]]:
    # Preparation of the dataloaders
    train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict = prepare_dataloaders(train_val_data, local_test_data, new_test_data, params)

    # Initialization of the models
    global_model, models = init_federated_models(train_dls, params, architecture=SimpleAutoencoder)
    global_threshold = Threshold(torch.tensor(0.))

    # Initialization of the results
    local_results, new_devices_results, global_thresholds = [], [], []

    # Selection of a client to mimic in case we use the mimic attack
    mimicked_client_id = select_mimicked_client(params)

    for federation_round in range(params.federation_rounds):
        print_federation_round(federation_round, params.federation_rounds)

        # Local training of each client
        multitrain_autoencoders(trains=list(zip(['Training client {} on: '.format(i) + device_names(client_devices)
                                                 for i, client_devices in enumerate(params.clients_devices)],
                                                train_dls, models)),
                                params=params, lr_factor=(params.gamma_round ** federation_round),
                                main_title='Training the clients', color=Color.GREEN)

        # Model poisoning attacks
        models = model_poisoning(global_model, models, params, mimicked_client_id=mimicked_client_id, verbose=True)

        # Aggregation
        global_model, models = model_aggregation(global_model, models, params, verbose=True)

        # Compute and aggregate thresholds
        federated_thresholds(models, threshold_dls, global_threshold, params, global_thresholds)

        # Testing
        federated_testing(global_model, global_threshold, local_test_dls_dicts, new_test_dls_dict, params, local_results, new_devices_results)

        Ctp.exit_section()

    return local_results, new_devices_results, global_thresholds


def fedsgd_autoencoders_train_test(train_val_data: FederationData, local_test_data: FederationData,
                                   new_test_data: ClientData, params: SimpleNamespace)\
        -> Tuple[List[BinaryClassificationResult], List[BinaryClassificationResult], List[float]]:
    # Preparation of the dataloaders
    train_dls, threshold_dls, local_test_dls_dicts, new_test_dls_dict = prepare_dataloaders(train_val_data, local_test_data, new_test_data, params)

    # Initialization of the models
    global_model, models = init_federated_models(train_dls, params, architecture=SimpleAutoencoder)
    global_threshold = Threshold(torch.tensor(0.))

    # Initialization of the results
    local_results, new_devices_results, global_thresholds = [], [], []

    # Selection of a client to mimic in case we use the mimic attack
    mimicked_client_id = select_mimicked_client(params)

    for epoch in range(params.epochs):
        print_federation_epoch(epoch, params.epochs)
        lr_factor = params.lr_scheduler_params['gamma'] ** (epoch // params.lr_scheduler_params['step_size'])
        global_model, models = train_autoencoders_fedsgd(global_model, models, train_dls, params, lr_factor=lr_factor,
                                                         mimicked_client_id=mimicked_client_id)

        if epoch % 10 == 0:
            # Compute and aggregate thresholds
            federated_thresholds(models, threshold_dls, global_threshold, params, global_thresholds)

            # Testing
            federated_testing(global_model, global_threshold, local_test_dls_dicts, new_test_dls_dict, params, local_results, new_devices_results)
        Ctp.exit_section()

    return local_results, new_devices_results, global_thresholds
