import itertools
import sys
from types import SimpleNamespace

import torch
import torch.utils.data
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from anomaly_detection_experiments import local_autoencoders, federated_autoencoders
from classification_experiments import local_classifiers, federated_classifiers
from data import get_all_data, split_data, split_data_current_fold
from saving import save_results, create_dirs, save_dummy_results

from typing import List, Dict, Callable
import numpy as np


def run_grid_search(train_val_data: List[Dict[str, np.array]], experiment_function: Callable,
                    constant_args: dict, varying_args: dict, configurations: List[Dict[str, list]], n_folds: int = 1) -> None:

    # Create the path in which we store the results and try to store a dummy result file here
    # (if it does not work it's better to know before we do the whole grid search)
    results_path = 'grid_search_results/' + experiment_function.__name__ + '/'
    create_dirs(results_path)
    save_dummy_results(results_path)

    args_dict = constant_args
    product = list(itertools.product(*varying_args.values()))  # Compute the different sets of hyper-parameters to test in the grid search
    local_results, new_devices_results = {}, {}

    for i, experiment_args_tuple in enumerate(product):  # Grid search: we iterate over the sets of parameters to be tested
        experiment_args = {key: arg for (key, arg) in zip(varying_args.keys(), experiment_args_tuple)}
        args_dict.update(experiment_args)
        Ctp.enter_section('Experiment [{}/{}] with args: '.format(i + 1, len(product)) + str(experiment_args), Color.WHITE)
        local_results[repr(experiment_args)] = {}
        new_devices_results[repr(experiment_args)] = {}
        for j, configuration in enumerate(configurations):  # Multiple configurations: we iterate over the possible configurations of the clients
            Ctp.enter_section('Configuration [{}/{}]: '.format(j + 1, len(configurations)) + str(configuration), Color.NONE)
            args_dict.update(configuration)
            args = SimpleNamespace(**args_dict)
            local_results[repr(experiment_args)][repr(configuration)] = []
            new_devices_results[repr(experiment_args)][repr(configuration)] = []
            if n_folds == 1:  # We do not use cross-validation
                train_data, val_data = split_data(train_val_data, p_test=0.2, p_unused=0.0)
                local_result, new_devices_result = experiment_function(train_data, val_data, args=args)
                local_results[repr(experiment_args)][repr(configuration)] = [local_result]
                new_devices_results[repr(experiment_args)][repr(configuration)] = [new_devices_result]
            else:
                for fold in range(n_folds):  # Cross validation: we iterate over the folds
                    Ctp.enter_section('Fold [{}/{}]'.format(fold + 1, n_folds), Color.GRAY)
                    train_data, val_data = split_data_current_fold(train_val_data, n_folds, fold)
                    local_result, new_devices_result = experiment_function(train_data, val_data, args=args)
                    local_results[repr(experiment_args)][repr(configuration)].append(local_result)
                    new_devices_results[repr(experiment_args)][repr(configuration)].append(new_devices_result)
                    Ctp.exit_section()
            Ctp.exit_section()
        Ctp.exit_section()

    save_results(results_path, local_results, new_devices_results)


def test_parameters(train_data: List[Dict[str, np.array]], test_data: List[Dict[str, np.array]],
                    experiment_function: Callable, args_dict: dict, configurations: List[Dict[str, list]],
                    n_random_reruns: int = 5) -> None:

    # Create the path in which we store the results and try to store a dummy result file here
    # (if it does not work it's better to know before we do the whole testing)
    results_path = 'test_results/' + experiment_function.__name__ + '/'
    create_dirs(results_path)
    save_dummy_results(results_path)

    local_results, new_devices_results = {}, {}

    for j, configuration in enumerate(configurations):  # Multiple configurations: we iterate over the possible configurations of the clients
        Ctp.enter_section('Configuration [{}/{}]: '.format(j + 1, len(configurations)) + str(configuration), Color.NONE)
        local_results[repr(configuration)] = []
        new_devices_results[repr(configuration)] = []
        for run_id in range(n_random_reruns):
            local_result, new_devices_result = experiment_function(train_data, test_data, args=SimpleNamespace(**args_dict))
            local_results[repr(configuration)].append(local_result)
            new_devices_results[repr(configuration)].append(new_devices_result)

    save_results(results_path, local_results, new_devices_results)


def main(experiment: str = 'single_classifier'):
    Ctp.set_max_depth(4)
    Ctp.set_automatic_skip(True)

    # Loading the data
    data = get_all_data(Color.YELLOW)
    train_val_data, test_data = split_data(data, p_test=0.2, p_unused=0.01)

    common_params = {'n_features': 115,
                     'normalization': 'min-max',
                     'test_bs': 4096}

    autoencoder_params = {'hidden_layers': [86, 58, 38, 29, 38, 58, 86],
                          'activation_fn': torch.nn.ELU}

    classifier_params = {'hidden_layers': [40, 10, 5],
                         'activation_fn': torch.nn.ELU}

    decentralized_configurations = [{'clients_devices': [[0], [1], [2], [3], [4], [5], [6], [7]], 'test_devices': [8]},
                                    {'clients_devices': [[0], [1], [2], [3], [4], [5], [6], [8]], 'test_devices': [7]},
                                    {'clients_devices': [[0], [1], [2], [3], [4], [5], [7], [8]], 'test_devices': [6]},
                                    {'clients_devices': [[0], [1], [2], [3], [4], [6], [7], [8]], 'test_devices': [5]},
                                    {'clients_devices': [[0], [1], [2], [3], [5], [6], [7], [8]], 'test_devices': [4]},
                                    {'clients_devices': [[0], [1], [2], [4], [5], [6], [7], [8]], 'test_devices': [3]},
                                    {'clients_devices': [[0], [1], [3], [4], [5], [6], [7], [8]], 'test_devices': [2]},
                                    {'clients_devices': [[0], [2], [3], [4], [5], [6], [7], [8]], 'test_devices': [1]},
                                    {'clients_devices': [[1], [2], [3], [4], [5], [6], [7], [8]], 'test_devices': [0]}]

    centralized_configurations = [{'clients_devices': [[0, 1, 2, 3, 4, 5, 6, 7]], 'test_devices': [8]},
                                  {'clients_devices': [[0, 1, 2, 3, 4, 5, 6, 8]], 'test_devices': [7]},
                                  {'clients_devices': [[0, 1, 2, 3, 4, 5, 7, 8]], 'test_devices': [6]},
                                  {'clients_devices': [[0, 1, 2, 3, 4, 6, 7, 8]], 'test_devices': [5]},
                                  {'clients_devices': [[0, 1, 2, 3, 5, 6, 7, 8]], 'test_devices': [4]},
                                  {'clients_devices': [[0, 1, 2, 4, 5, 6, 7, 8]], 'test_devices': [3]},
                                  {'clients_devices': [[0, 1, 3, 4, 5, 6, 7, 8]], 'test_devices': [2]},
                                  {'clients_devices': [[0, 2, 3, 4, 5, 6, 7, 8]], 'test_devices': [1]},
                                  {'clients_devices': [[1, 2, 3, 4, 5, 6, 7, 8]], 'test_devices': [0]}]

    autoencoder_opt_default_params = {'epochs': 1,
                                      'train_bs': 64,
                                      'optimizer': torch.optim.Adadelta,
                                      'optimizer_params': {'lr': 1.0, 'weight_decay': 5 * 1e-5},
                                      'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      'lr_scheduler_params': {'patience': 3, 'threshold': 1e-2, 'factor': 0.5, 'verbose': False}}

    autoencoder_opt_federated_params = {'epochs': 30,
                                        'train_bs': 64,
                                        'optimizer': torch.optim.Adadelta,
                                        'optimizer_params': {'lr': 1.0, 'weight_decay': 5 * 1e-5},
                                        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                        'lr_scheduler_params': {'step_size': 1, 'gamma': 0.9},
                                        'federation_rounds': 3,
                                        'gamma_round': 0.5}

    classifier_opt_default_params = {'epochs': 1,
                                     'train_bs': 64,
                                     'optimizer': torch.optim.Adadelta,
                                     'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-5},
                                     'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                     'lr_scheduler_params': {'step_size': 1, 'gamma': 0.5}}

    classifier_opt_federated_params = {'epochs': 1,
                                       'train_bs': 64,
                                       'optimizer': torch.optim.Adadelta,
                                       'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-5},
                                       'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                       'lr_scheduler_params': {'step_size': 1, 'gamma': 0.5},
                                       'federation_rounds': 1,
                                       'gamma_round': 0.5}

    if experiment == 'single_autoencoder':
        Ctp.print('\n\t\t\t\t\tSINGLE AUTOENCODER GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_autoencoders,
                        {**common_params, **autoencoder_params, **autoencoder_opt_default_params},
                        {'normalization': ['0-mean 1-var', 'min-max'], 'hidden_layers': [[20, 5, 20], [86, 58, 38, 29, 38, 58, 86]]},
                        centralized_configurations, n_folds=1)

    elif experiment == 'multiple_autoencoders':
        Ctp.print('\n\t\t\t\t\tMULTIPLE AUTOENCODERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_autoencoders,
                        {**common_params, **autoencoder_params, **autoencoder_opt_default_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations, n_folds=1)

    elif experiment == 'federated_autoencoders':
        Ctp.print('\n\t\t\t\t\tFEDERATED AUTOENCODERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, federated_autoencoders,
                        {**common_params, **autoencoder_params, **autoencoder_opt_federated_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations, n_folds=1)

    elif experiment == 'single_classifier':
        Ctp.print('\n\t\t\t\t\tSINGLE CLASSIFIER GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_classifiers,
                        {**common_params, **classifier_params, **classifier_opt_default_params},
                        {'normalization': ['min-max']},
                        centralized_configurations, n_folds=1)

    elif experiment == 'multiple_classifiers':
        Ctp.print('\n\t\t\t\t\tMULTIPLE CLASSIFIERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_classifiers,
                        {**common_params, **classifier_params, **classifier_opt_default_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations, n_folds=1)

    elif experiment == 'federated_classifiers':
        Ctp.print('\n\t\t\t\t\tFEDERATED CLASSIFIERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, federated_classifiers,
                        {**common_params, **classifier_params, **classifier_opt_federated_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations[:1], n_folds=1)

# TODO: better type hinting for the whole project

# TODO: rethink of the way the results should be stored
#  grid search case: results/grid_search/run_0/(local_results.json, new_devices_results.json, fixed_parameters.json)
#  the run id should be automatically implemented based on existing folders
#  test case: results/test/run_0/(local_results.json, new_devices_results.json, fixed_parameters.json)
#  2 things to code:
#  - be able to store any set of parameters as json
#  - be able to automatically find the id of the run in order to name the folder properly


if __name__ == "__main__":
    main(sys.argv[1])
