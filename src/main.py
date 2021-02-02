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


def run_grid_search(train_val_data, experiment_function, constant_args: dict, varying_args: dict, configurations: list, n_folds=5):
    args_dict = constant_args
    product = list(itertools.product(*varying_args.values()))
    for i, experiment_args_tuple in enumerate(product):
        experiment_args = {key: arg for (key, arg) in zip(varying_args.keys(), experiment_args_tuple)}
        args_dict.update(experiment_args)
        Ctp.enter_section('Experiment [{}/{}] with args: '.format(i + 1, len(product)) + str(experiment_args), Color.WHITE)
        for j, configuration in enumerate(configurations):
            Ctp.enter_section('Configuration [{}/{}]: '.format(j + 1, len(configurations)) + str(configuration), Color.NONE)
            args_dict.update(configuration)
            args = SimpleNamespace(**args_dict)
            if n_folds == 1:  # We do not use cross-validation
                train_data, val_data = split_data(train_val_data, p_test=0.2, p_unused=0.0)
                experiment_function(train_data, val_data, args=args)
            else:
                for fold in range(n_folds):  # Cross validation
                    Ctp.enter_section('Fold [{}/{}]'.format(fold + 1, n_folds), Color.GRAY)
                    train_data, val_data = split_data_current_fold(train_val_data, n_folds, fold)
                    experiment_function(train_data, val_data, args=args)
                    Ctp.exit_section()
            Ctp.exit_section()
        Ctp.exit_section()


def test_parameters(train_data, test_data, experiment_function, args_dict, n_random_reruns=5):
    for run_id in range(n_random_reruns):
        # Load train/test data
        dataloaders = None
        experiment_function(dataloaders, SimpleNamespace(**args_dict))


def main(experiment='single_classifier'):
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
                                       'federation_rounds': 5,
                                       'gamma_round': 0.5}

    if experiment == 'single_autoencoder':
        Ctp.print('\n\t\t\t\t\tSINGLE AUTOENCODER GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_autoencoders,
                        {**common_params, **autoencoder_params, **autoencoder_opt_default_params},
                        {'normalization': ['0-mean 1-var', 'min-max'], 'hidden_layers': [[20, 5, 20], [86, 58, 38, 29, 38, 58, 86]]},
                        centralized_configurations)

    elif experiment == 'multiple_autoencoders':
        Ctp.print('\n\t\t\t\t\tMULTIPLE AUTOENCODERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_autoencoders,
                        {**common_params, **autoencoder_params, **autoencoder_opt_default_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations)

    elif experiment == 'federated_autoencoders':
        Ctp.print('\n\t\t\t\t\tFEDERATED AUTOENCODERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, federated_autoencoders,
                        {**common_params, **autoencoder_params, **autoencoder_opt_federated_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations)

    elif experiment == 'single_classifier':
        Ctp.print('\n\t\t\t\t\tSINGLE CLASSIFIER GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_classifiers,
                        {**common_params, **classifier_params, **classifier_opt_default_params},
                        {'normalization': ['min-max']},
                        centralized_configurations)

    elif experiment == 'multiple_classifiers':
        Ctp.print('\n\t\t\t\t\tMULTIPLE CLASSIFIERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, local_classifiers,
                        {**common_params, **classifier_params, **classifier_opt_default_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations)

    elif experiment == 'federated_classifiers':
        Ctp.print('\n\t\t\t\t\tFEDERATED CLASSIFIERS GRID SEARCH\n', bold=True)
        run_grid_search(train_val_data, federated_classifiers,
                        {**common_params, **classifier_params, **classifier_opt_federated_params},
                        {'normalization': ['min-max']},
                        decentralized_configurations)


# TODO: each experiment function should return some results to be able to know with which arguments it performed the best

# TODO: make a test function that should use the test set only and test a specific set of parameters

if __name__ == "__main__":
    main(sys.argv[1])
