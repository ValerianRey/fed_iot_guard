from argparse import ArgumentParser

import torch
import torch.utils.data
from context_printer import ContextPrinter as Ctp

from data import read_all_data, all_devices
from grid_search import run_grid_search
from supervised_data import get_client_supervised_initial_splitting
from supervised_experiments import local_classifiers_train_test, federated_classifiers_train_test, local_classifier_train_val
from test_hparams import test_hyperparameters
from unsupervised_data import get_client_unsupervised_initial_splitting
from unsupervised_experiments import local_autoencoder_train_val, local_autoencoders_train_test, federated_autoencoders_train_test


def main(experiment: str, setup: str, federated: bool, test: bool):
    Ctp.set_automatic_skip(True)
    Ctp.print('\n\t\t\t\t\t' + ('FEDERATED ' if federated else '') + setup.upper() + ' ' + experiment.upper()
              + (' TESTING' if test else ' GRID SEARCH') + '\n', bold=True)

    common_params = {'n_features': 115,
                     'normalization': 'min-max',
                     'test_bs': 4096}

    autoencoder_params = {'hidden_layers': [29],
                          'activation_fn': torch.nn.ELU,
                          'p_threshold': 0.5}  # The proportion of benign data that we use to compute the threshold

    classifier_params = {'hidden_layers': [40, 10, 5],
                         'activation_fn': torch.nn.ELU}

    n_devices = len(all_devices)

    # 9 configurations in which we have 8 clients (each one owns the data from 1 device) and he data from the last device is left unseen.
    decentralized_configurations = [{'clients_devices': [[i] for i in range(n_devices) if i != test_device],
                                     'test_devices': [test_device]} for test_device in range(n_devices)]

    # 9 configurations in which we have 1 client (owning the data from 8 devices) and he data from the last device is left unseen.
    centralized_configurations = [{'clients_devices': [[i for i in range(n_devices) if i != test_device]],
                                   'test_devices': [test_device]} for test_device in range(n_devices)]

    autoencoder_opt_default_params = {'epochs': 0,  # 50
                                      'train_bs': 64,
                                      'optimizer': torch.optim.Adadelta,
                                      'optimizer_params': {'lr': 1.0, 'weight_decay': 5 * 1e-5},
                                      'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      'lr_scheduler_params': {'patience': 3, 'threshold': 1e-2, 'factor': 0.5, 'verbose': False}}

    classifier_opt_default_params = {'epochs': 0,  # 4
                                     'train_bs': 64,
                                     'optimizer': torch.optim.Adadelta,
                                     'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-5},
                                     'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                     'lr_scheduler_params': {'step_size': 1, 'gamma': 0.5}}

    federation_params = {'federation_rounds': 2, 'gamma_round': 0.75}

    # Loading the data
    all_data = read_all_data()
    p_test = 0.2
    p_unused = 0.01
    p_val = 0.3
    n_splits = 1
    n_random_reruns = 1

    if setup == 'centralized':
        configurations = centralized_configurations
    elif setup == 'decentralized':
        configurations = decentralized_configurations
    else:
        raise ValueError
    name = setup + '_' + experiment

    if experiment == 'autoencoder':
        constant_params = {**common_params, **autoencoder_params, **autoencoder_opt_default_params}
        splitting_function = get_client_unsupervised_initial_splitting
        if test:
            if federated:
                constant_params.update(federation_params)
                test_function = federated_autoencoders_train_test
                name += '_federated'
            else:
                test_function = local_autoencoders_train_test

            configurations_params = [{} for _ in range(len(configurations))]
            # set the hyper-parameters specific to each configuration (overrides the parameters defined in constant_params)

            test_hyperparameters(all_data, name, test_function, splitting_function, constant_params, configurations_params, configurations,
                                 p_test=p_test, p_unused=p_unused, n_random_reruns=n_random_reruns)
        else:
            varying_params = {'normalization': ['0-mean 1-var', 'min-max'],
                              'hidden_layers': [[11], [38, 11, 38], [58, 38, 29, 11, 29, 38, 58], [29], [58, 29, 58], [86, 58, 38, 29, 38, 58, 86]]}
            run_grid_search(all_data, name, local_autoencoder_train_val, splitting_function, constant_params, varying_params, configurations,
                            p_test=p_test, p_unused=p_unused, n_splits=n_splits, p_val=p_val)

    elif experiment == 'classifier':
        constant_params = {**common_params, **classifier_params, **classifier_opt_default_params}
        splitting_function = get_client_supervised_initial_splitting

        if test:
            if federated:
                constant_params.update(federation_params)
                test_function = federated_classifiers_train_test
                name += '_federated'
            else:
                test_function = local_classifiers_train_test

            configurations_params = [{} for _ in range(len(configurations))]
            # set the hyper-parameters specific to each configuration (overrides the parameters defined in constant_params)

            test_hyperparameters(all_data, name, test_function, splitting_function, constant_params, configurations_params, configurations,
                                 p_test=p_test, p_unused=p_unused, n_random_reruns=n_random_reruns)
        else:
            varying_params = {'normalization': ['0-mean 1-var', 'min-max'],
                              'optimizer_params': [{'lr': 1.0, 'weight_decay': 1e-5}, {'lr': 1.0, 'weight_decay': 5 * 1e-5}]}
            run_grid_search(all_data, name, local_classifier_train_val, splitting_function, constant_params, varying_params, configurations,
                            p_test=p_test, p_unused=p_unused, n_splits=n_splits, p_val=p_val)
    else:
        raise ValueError


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('setup', help='centralized or decentralized')
    parser.add_argument('experiment', help='Experiment to run (classifier or autoencoder)')

    test_parser = parser.add_mutually_exclusive_group(required=False)
    test_parser.add_argument('--test', dest='test', action='store_true')
    test_parser.add_argument('--gs', dest='test', action='store_false')
    parser.set_defaults(test=False)

    federated_parser = parser.add_mutually_exclusive_group(required=False)
    federated_parser.add_argument('--federated', dest='federated', action='store_true')
    federated_parser.add_argument('--no-federated', dest='federated', action='store_false')
    parser.set_defaults(federated=False)

    verbose_parser = parser.add_mutually_exclusive_group(required=False)
    verbose_parser.add_argument('--verbose', dest='verbose', action='store_true')
    verbose_parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    parser.add_argument('--verbose-depth', dest='max_depth', type=int, help='Maximum number of nested sections after which the printing will stop')
    parser.set_defaults(max_depth=None)

    args = parser.parse_args()

    if not args.verbose:  # Deactivate all printing in the console
        Ctp.deactivate()

    if args.max_depth is not None:
        Ctp.set_max_depth(args.max_depth)  # Set the max depth at which we print in the console

    main(args.experiment, args.setup, args.federated, args.test)

# TODO: (re)implement notebook to analyse grid search results:
#  try out every gs results with dummy results

