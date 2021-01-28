import sys
from types import SimpleNamespace

import torch
import torch.utils.data

from classification_experiments import local_classifiers, federated_classifiers
from anomaly_detection_experiments import local_autoencoders, federated_autoencoders


def main(experiment='single_classifier'):
    common_params = {'n_features': 115,
                     'normalization': 'min-max',
                     'test_bs': 4096}

    autoencoder_params = {'hidden_layers': [86, 58, 38, 29, 38, 58, 86],
                          'activation_fn': torch.nn.ELU}

    classifier_params = {'hidden_layers': [40, 10, 5],
                         'activation_fn': torch.nn.ELU}

    multiple_clients_params = {'clients_devices': [[0], [1], [2], [3], [4], [5], [6], [7]],
                               'test_devices': [8]}

    single_client_params = {'clients_devices': [[0, 1, 2, 3, 4, 5, 6, 7]],
                            'test_devices': [8]}

    autoencoder_opt_default_params = {'epochs': 20,
                                      'train_bs': 64,
                                      'optimizer': torch.optim.Adadelta,
                                      'optimizer_params': {'lr': 1.0, 'weight_decay': 5 * 1e-5},
                                      'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      'lr_scheduler_params': {'patience': 3, 'threshold': 1e-2, 'factor': 0.5, 'verbose': False}}

    autoencoder_opt_federated_params = {'epochs': 20,
                                        'train_bs': 64,
                                        'optimizer': torch.optim.Adadelta,
                                        'optimizer_params': {'lr': 1.0, 'weight_decay': 5 * 1e-5},
                                        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                        'lr_scheduler_params': {'step_size': 1, 'gamma': 0.9},
                                        'federation_rounds': 3,
                                        'gamma_round': 0.5}

    classifier_opt_default_params = {'epochs': 3,
                                     'train_bs': 64,
                                     'optimizer': torch.optim.Adadelta,
                                     'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-5},
                                     'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                     'lr_scheduler_params': {'step_size': 1, 'gamma': 0.5}}

    classifier_opt_federated_params = {'epochs': 3,
                                       'train_bs': 64,
                                       'optimizer': torch.optim.Adadelta,
                                       'optimizer_params': {'lr': 1.0, 'weight_decay': 1e-5},
                                       'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                       'lr_scheduler_params': {'step_size': 1, 'gamma': 0.5},
                                       'federation_rounds': 5,
                                       'gamma_round': 0.5}

    if experiment == 'single_autoencoder':
        local_autoencoders(args=SimpleNamespace(**common_params, **autoencoder_params,
                                                **autoencoder_opt_default_params, **single_client_params))

    elif experiment == 'multiple_autoencoders':
        local_autoencoders(args=SimpleNamespace(**common_params, **autoencoder_params,
                                                **autoencoder_opt_default_params, **multiple_clients_params))

    elif experiment == 'federated_autoencoders':
        federated_autoencoders(args=SimpleNamespace(**common_params, **autoencoder_params,
                                                    **autoencoder_opt_federated_params, **multiple_clients_params))

    elif experiment == 'single_classifier':
        local_classifiers(args=SimpleNamespace(**common_params, **classifier_params,
                                               **classifier_opt_default_params, **single_client_params))

    elif experiment == 'multiple_classifiers':
        local_classifiers(args=SimpleNamespace(**common_params, **classifier_params,
                                               **classifier_opt_default_params, **multiple_clients_params))

    elif experiment == 'federated_classifiers':
        federated_classifiers(args=SimpleNamespace(**common_params, **classifier_params,
                                                   **classifier_opt_federated_params, **multiple_clients_params))


if __name__ == "__main__":
    main(sys.argv[1])
