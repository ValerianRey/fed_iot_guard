from argparse import ArgumentParser

import torch.utils.data
from context_printer import ContextPrinter as Ctp

from data import read_all_data, all_devices
from federated_util import *
from grid_search import run_grid_search
from supervised_data import get_client_supervised_initial_splitting
from test_hparams import test_hyperparameters
from unsupervised_data import get_client_unsupervised_initial_splitting


def main(experiment: str, setup: str, federated: bool, test: bool):
    Ctp.set_automatic_skip(True)
    Ctp.print('\n\t\t\t\t\t' + ('FEDERATED ' if federated else '') + setup.upper() + ' ' + experiment.upper()
              + (' TESTING' if test else ' GRID SEARCH') + '\n', bold=True)

    common_params = {'n_features': 115,
                     'normalization': 'min-max',
                     'test_bs': 4096,
                     'p_test': 0.2,
                     'p_unused': 0.01,
                     'val_part': None,  # This is the proportion of training data that goes into the validation set, not the proportion of all data
                     'n_splits': 5,
                     'n_random_reruns': 1,
                     'cuda': False,  # It looks like cuda is slower than CPU for me so I enforce using the CPU
                     'benign_prop': 0.95,
                     # Desired proportion of benign data in the train/validation sets (or None to keep the natural proportions)
                     'samples_per_device': 10_000}  # Total number of datapoints (train & val + unused + test) for each device.

    # p_test, p_unused and p_train_val are the proportions of *all data* that go into respectively the test set, the unused set and the train &
    # validation set.
    # val_part and threshold_part are the proportions of the the *train & validation set* used for respectively the validation and the threshold
    # note that we either use one or the other: when grid searching we do not compute the threshold, so we have the train & validation set
    # split between val_part proportion of validation data and (1. - val_part) proportion of train data
    # when testing hyper-params, we have train & validation set split between threshold_part proportion of threshold data and
    # (1. - threshold_part) proportion of train data.
    # benign_prop is yet another thing, determining the proportion of benign data when applicable (everywhere except in the train_val set of
    # the unsupervised method)

    p_train_val = 1. - common_params['p_test'] - common_params['p_unused']

    if common_params['val_part'] is None:
        val_part = 1. / common_params['n_splits']
    else:
        val_part = common_params['val_part']

    common_params.update({'p_train_val': p_train_val, 'val_part': val_part})

    if common_params['cuda']:
        Ctp.print('Using CUDA')
    else:
        Ctp.print('Using CPU')

    autoencoder_params = {'hidden_layers': [29],
                          'activation_fn': torch.nn.ELU,
                          'threshold_part': 0.5,
                          'quantile': 0.95}

    classifier_params = {'hidden_layers': [115, 58, 29],
                         'activation_fn': torch.nn.ELU}

    n_devices = len(all_devices)

    # 9 configurations in which we have 8 clients (each one owns the data from 1 device) and he data from the last device is left unseen.
    decentralized_configurations = [{'clients_devices': [[i] for i in range(n_devices) if i != test_device],
                                     'test_devices': [test_device]} for test_device in range(n_devices)]

    # 9 configurations in which we have 1 client (owning the data from 8 devices) and he data from the last device is left unseen.
    centralized_configurations = [{'clients_devices': [[i for i in range(n_devices) if i != test_device]],
                                   'test_devices': [test_device]} for test_device in range(n_devices)]

    autoencoder_opt_default_params = {'epochs': 120,
                                      'train_bs': 64,
                                      'optimizer': torch.optim.SGD,
                                      'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-5},
                                      'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                      'lr_scheduler_params': {'step_size': 20, 'gamma': 0.5}}
                                      # 'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      # 'lr_scheduler_params': {'patience': 3, 'threshold': 0.025, 'factor': 0.5, 'verbose': False}}

    classifier_opt_default_params = {'epochs': 4,
                                     'train_bs': 64,

                                     'optimizer': torch.optim.SGD,
                                     'optimizer_params': {'lr': 0.5, 'weight_decay': 1e-5},
                                     'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                     'lr_scheduler_params': {'step_size': 1, 'gamma': 0.5}}

    federation_params = {'federation_rounds': 10, 'gamma_round': 0.75, 'aggregation_function': federated_averaging,
                         'resampling': None}

    # data_poisoning: 'all_labels_flipping', 'benign_labels_flipping', 'attack_labels_flipping'
    # model_poisoning: 'cancel_attack', 'mimic_attack'
    # model update factor is the factor by which the difference between the original (global) model and the trained model is multiplied
    # (only applies to the malicious clients; for honest clients this factor is always 1)
    poisoning_params = {'n_malicious': 0, 'data_poisoning': None, 'p_poison': None,
                        'model_update_factor': 1.0, 'model_poisoning': None}

    # Loading the data
    all_data = read_all_data()

    if setup == 'centralized':
        configurations = centralized_configurations
    elif setup == 'decentralized':
        configurations = decentralized_configurations
    else:
        raise ValueError

    Ctp.print(configurations)

    if experiment == 'autoencoder':
        constant_params = {**common_params, **autoencoder_params, **autoencoder_opt_default_params}
        splitting_function = get_client_unsupervised_initial_splitting
        if test:
            if federated:
                constant_params.update(federation_params)

            # set the hyper-parameters specific to each configuration (overrides the parameters defined in constant_params)
            configurations_params = [{'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-05}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}},
                                     {'hidden_layers': [29], 'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}}]

            test_hyperparameters(all_data, setup, experiment, federated, splitting_function, constant_params, configurations_params, configurations)
        else:
            varying_params = {'hidden_layers': [[86, 58, 38, 29, 38, 58, 86], [58, 29, 58], [29]],
                              'optimizer_params': [{'lr': 1.0, 'weight_decay': 0.},
                                                   {'lr': 1.0, 'weight_decay': 1e-5},
                                                   {'lr': 1.0, 'weight_decay': 1e-4}]}
            run_grid_search(all_data, setup, experiment, splitting_function, constant_params, varying_params, configurations)

    elif experiment == 'classifier':
        constant_params = {**common_params, **classifier_params, **classifier_opt_default_params, **poisoning_params}
        splitting_function = get_client_supervised_initial_splitting

        if test:
            if federated:
                constant_params.update(federation_params)

            # set the hyper-parameters specific to each configuration (overrides the parameters defined in constant_params)
            configurations_params = [{'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}, 'hidden_layers': [115, 58, 29]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}, 'hidden_layers': [115, 58, 29]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}, 'hidden_layers': [115, 58, 29]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}, 'hidden_layers': [115, 58, 29]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}, 'hidden_layers': [115, 58]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-05}, 'hidden_layers': [115, 58, 29]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}, 'hidden_layers': [115, 58, 29]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-05}, 'hidden_layers': [115, 58, 29]},
                                     {'optimizer_params': {'lr': 1.0, 'weight_decay': 0.0}, 'hidden_layers': [115, 58, 29]}]

            test_hyperparameters(all_data, setup, experiment, federated, splitting_function, constant_params, configurations_params, configurations)
        else:
            varying_params = {'optimizer_params': [{'lr': 0.5, 'weight_decay': 0.},
                                                   {'lr': 0.5, 'weight_decay': 1e-5},
                                                   {'lr': 0.5, 'weight_decay': 1e-4}],
                              'hidden_layers': [[115, 58, 29], [115, 58], [115], []]}
            run_grid_search(all_data, setup, experiment, splitting_function, constant_params, varying_params, configurations)
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
