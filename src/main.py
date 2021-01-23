import sys
from types import SimpleNamespace

import torch
import torch.utils.data

from classification_experiments import single_classifier, multiple_classifiers, federated_classifiers
from src.anomaly_detection_experiments import single_autoencoder, multiple_autoencoders, federated_autoencoders


def main(experiment='single_classifier'):
    common_params = {'normalization': '0-mean 1-var',
                     'test_bs': 4096}

    autoencoder_params = {'hidden_layers': [86, 58, 38, 29, 38, 58, 86],
                          'activation_fn': torch.nn.ELU}

    classifier_params = {'hidden_layers': [40, 10, 5],
                         'activation_fn': torch.nn.ELU}

    autoencoder_opt_default_params = {'epochs': 0,
                                      'train_bs': 64,
                                      'optimizer': torch.optim.Adadelta,
                                      'optimizer_params': {'lr': 1.0, 'weight_decay': 5 * 1e-5},
                                      'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
                                      'lr_scheduler_params': {'patience': 5, 'threshold': 1e-2, 'factor': 0.5, 'verbose': False}}

    autoencoder_opt_federated_params = {'epochs': 20,
                                        'train_bs': 64,
                                        'optimizer': torch.optim.Adadelta,
                                        'optimizer_params': {'lr': 1.0, 'weight_decay': 5 * 1e-5},
                                        'lr_scheduler': torch.optim.lr_scheduler.StepLR,
                                        'lr_scheduler_params': {'step_size': 1, 'gamma': 0.9},
                                        'federation_rounds': 3,
                                        'gamma_round': 0.5}

    classifier_opt_default_params = {'epochs': 2,
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
                                       'federation_rounds': 3,
                                       'gamma_round': 0.5}

    if experiment == 'single_autoencoder':
        single_autoencoder(args=SimpleNamespace(**common_params, **autoencoder_params, **autoencoder_opt_default_params))

    elif experiment == 'multiple_autoencoders':
        multiple_autoencoders(args=SimpleNamespace(**common_params, **autoencoder_params, **autoencoder_opt_default_params))

    elif experiment == 'federated_autoencoders':
        federated_autoencoders(args=SimpleNamespace(**common_params, **autoencoder_params, **autoencoder_opt_federated_params))

    elif experiment == 'single_classifier':
        single_classifier(args=SimpleNamespace(**common_params, **classifier_params, **classifier_opt_default_params))

    elif experiment == 'multiple_classifiers':
        multiple_classifiers(args=SimpleNamespace(**common_params, **classifier_params, **classifier_opt_default_params))

    elif experiment == 'federated_classifiers':
        federated_classifiers(args=SimpleNamespace(**common_params, **classifier_params, **classifier_opt_federated_params))


# TODO: other models (for example autoencoder + reconstruction of next sample, multi-class classifier)
#  => the objective is to have a greater variety of results

# TODO: custom loss function (more weight on false positives than false negatives for example) (+ maybe place the criterion used in the args)

# TODO: make a few interesting experiments: test global model on a dataset that has been seen during training (actually already done)

# TODO: automatic writing of the results to a file (using the context printer)

# TODO: saving of the model

# TODO: evaluation mode (just test a model without training it first)

# TODO: grid search mode to find some hyper parameters, using a validation set

# TODO: use other aggregation methods

# TODO: implement random reruns that return avg and std results to get a sense of confidence interval

# TODO: implement cross validation

# TODO: change lrs for autoencoder (especially federated or multiple)


if __name__ == "__main__":
    main(sys.argv[1])
