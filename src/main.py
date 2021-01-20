import torch
import torch.utils.data
import sys
from types import SimpleNamespace
from classification_experiments import single_classifier, multiple_classifiers, federated_classifiers
from src.anomaly_detection_experiments import single_autoencoder, multiple_autoencoders


def main(experiment='single_classifier'):
    if experiment == 'single_autoencoder':
        single_autoencoder()
    elif experiment == 'multiple_autoencoders':
        multiple_autoencoders()
    elif experiment == 'single_classifier':
        single_classifier(args=SimpleNamespace(**{
            'epochs': 2,
            'train_bs': 64,
            'test_bs': 4096,
            'normalization': '0-mean 1-var',
            'lr': 1.0,
            'hidden_layers': [40, 10, 5],
            'activation_fn': torch.nn.ELU
        }))
    elif experiment == 'multiple_classifiers':
        multiple_classifiers(args=SimpleNamespace(**{
            'epochs': 2,
            'train_bs': 64,
            'test_bs': 4096,
            'normalization': '0-mean 1-var',
            'lr': 1.0,
            'hidden_layers': [40, 10, 5],
            'activation_fn': torch.nn.ELU
        }))
    elif experiment == 'federated_classifiers':
        federated_classifiers(args=SimpleNamespace(**{
            'federation_rounds': 3,
            'lr': 1.0,
            'gamma_round': 0.5,
            'epochs': 3,
            'train_bs': 64,
            'test_bs': 4096,
            'normalization': '0-mean 1-var',
            'hidden_layers': [40, 10, 5],
            'activation_fn': torch.nn.ELU
        }))


# TODO: other models (for example autoencoder + reconstruction of next sample, multi-class classifier)
#  => the objective is to have a greater variety of results

# TODO: clean up code

# TODO: custom loss function (more weight on false positives than false negatives for example)

# TODO: better handling of the parameters: separate between global parameters, architecture-specific parameters, and experiment-specific parameters
#  add optimization parameters, add the parameters for the autoencoders

# TODO: make a few interesting experiments: test global model on a dataset that has been seen during training (actually already done)

# TODO: automatic writing of the results to a file

# TODO: saving of the model

# TODO: evaluation mode (just test a model without training it first)


if __name__ == "__main__":
    main(sys.argv[1])
