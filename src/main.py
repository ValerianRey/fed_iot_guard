import sys
from types import SimpleNamespace

import torch
import torch.utils.data

from classification_experiments import local_classifiers, federated_classifiers
from anomaly_detection_experiments import single_autoencoder, multiple_autoencoders, federated_autoencoders, local_autoencoders


def main(experiment='single_classifier'):
    common_params = {'normalization': '0-mean 1-var',
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
                                       'federation_rounds': 3,
                                       'gamma_round': 0.5}

    if experiment == 'single_autoencoder':
        local_autoencoders(args=SimpleNamespace(**common_params, **autoencoder_params,
                                                **autoencoder_opt_default_params, **single_client_params))

    elif experiment == 'multiple_autoencoders':
        multiple_autoencoders(args=SimpleNamespace(**common_params, **autoencoder_params,
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


# TODO: other models (for example autoencoder + reconstruction of next sample, multi-class classifier)
#  => the objective is to have a greater variety of results

# TODO: custom loss function (more weight on false positives than false negatives for example) (+ maybe place the criterion used in the args)

# TODO: make a few interesting experiments: test global model on a dataset that has been seen during training (actually already done)
#  make the current federated experiment for all possible unseen devices instead of just the last one
#  the devices that we give to each client and the devices that we leave unseen could be a parameter to the federated_classifiers function

# TODO: automatic writing of the results to a file (using the context printer)

# TODO: saving of the model

# TODO: evaluation mode (just test a model without training it first)

# TODO: grid search mode to find some hyper parameters, using a validation set

# TODO: use other aggregation methods

# TODO: implement random reruns that return avg and std results to get a sense of confidence interval

# TODO: implement cross validation
#  For cross validation it would be cool to have something that only needs to load the data once and then make a single dataloader that
#  could change stance between train/validation/test and current fold
#  for example for a 9-fold CV with 10% left over for testing purposes, test data would be the last 10%,
#  train fold 0 would yield the first 80% of the data, test fold 0 would yield the 80% to 90% of the data
#  train fold 1 would yield the 10% to 90% of the data, test fold 1 would yield the first 10% of the data
#  etc ...
#  The dataloader would actually only contain a single dataset though

# TODO: change normalization for classifier so that is also uses attack train data and not just benign train data
#  also change it so that each client remembers its normalization values, and the global model normalize the data that it sees with average
#  values

# TODO: update autoencoder code so that it works similarly as classifier

# TODO: make get_dataset functions only require a single device as input

# TODO: the autoencoder experiments should only have one dataloader to test, that should contain a separate dataset for each class
#  (benign, mirai 1, mirai 2, ..., gafgyt 1, ...) along with the name of each dataset and the positivity of each dataset

# TODO: it seems like the normalization of the data can play a huge factor in the accuracy (no proof for that yet but just a guess)
#  thus is would be cool to have a more advanced normalization with a common factor and bias that can learn

if __name__ == "__main__":
    main(sys.argv[1])
