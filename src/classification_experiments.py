from copy import deepcopy
from data import get_classifier_datasets, all_devices
from classification_ml import multitrain_classifiers, multitest_classifiers
from architectures import BinaryClassifier
import torch
import torch.utils.data
from print_util import Color, print_federation_round
from federated_util import federated_averaging


def single_classifier(args):
    # Initialize the model
    model = BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    # Loading the data
    dataset_train, dataset_test = get_classifier_datasets(all_devices, normalization=args.normalization)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_bs, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_bs)

    print(Color.BOLD + '\n\t\t\t\t\tSINGLE CLASSIFIER\n' + Color.END)

    # Training
    multitrain_classifiers(trains=[('with data from all devices', dataloader_train, model)], lr=args.lr, epochs=args.epochs,
                           main_title='Training the single model', color=Color.GREEN)

    # Testing
    multitest_classifiers(tests=[('Model trained on all devices', dataloader_test, model)],
                          main_title='Testing the single model', color=Color.BLUE)


def multiple_classifiers(args):
    # Initialization of the models
    models = [BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers) for _ in range(len(all_devices))]

    # Loading the data and creating the dataloaders
    datasets = [get_classifier_datasets([device], normalization=args.normalization) for device in all_devices]
    dataloaders_train = [torch.utils.data.DataLoader(dataset_train, batch_size=args.train_bs, shuffle=True) for (dataset_train, _) in datasets]
    dataloaders_test = [torch.utils.data.DataLoader(dataset_test, batch_size=args.test_bs) for (_, dataset_test) in datasets]

    print(Color.BOLD + '\n\t\t\t\t\tMULTIPLE CLASSIFIERS\n' + Color.END)

    # Local training of each client
    multitrain_classifiers(trains=zip(['with data from ' + device for device in all_devices], dataloaders_train, models),
                           lr=args.lr, epochs=args.epochs,
                           main_title='Training the different clients', color=Color.GREEN)

    # Local testing
    multitest_classifiers(tests=zip(['Model trained on dataset ' + device for device in all_devices], dataloaders_test, models),
                          main_title='Testing different clients on their own data', color=Color.BLUE)


def federated_classifiers(args):
    # Initialization of a global model
    global_model = BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    # Loading the data and creating the dataloaders (one per client)
    datasets = [get_classifier_datasets([device], normalization=args.normalization) for device in all_devices]
    dataloaders_train = [torch.utils.data.DataLoader(dataset_train, batch_size=args.train_bs, shuffle=True) for (dataset_train, _) in datasets[:8]]
    dataloaders_test = [torch.utils.data.DataLoader(dataset_test, batch_size=args.test_bs) for (_, dataset_test) in datasets]
    print()

    for federation_round in range(args.federation_rounds):
        print_federation_round(federation_round, args.federation_rounds)

        # Distribute the global model to all clients
        models = [deepcopy(global_model) for _ in all_devices[:8]]

        # Local training of each client
        multitrain_classifiers(trains=zip(['with data from ' + device for device in all_devices[:8]], dataloaders_train[:8], models[:8]),
                               lr=(args.lr * args.gamma_round ** federation_round), epochs=args.epochs,
                               main_title='Training the different clients', color=Color.GREEN)

        # Experiment 1: test all 8 trained clients on the data of the unseen device (9th)
        multitest_classifiers(tests=zip(['Model trained on dataset ' + device for device in all_devices[:8]],
                                        [dataloaders_test[8] for _ in range(8)],  # we always test on the same data
                                        models[:8]),
                              main_title='Testing different models on data from device ' + all_devices[8], color=Color.BLUE)

        # Experiment 2: test the global model on the data of all devices, before it has been averaged
        multitest_classifiers(tests=zip(['Data from device ' + device for device in all_devices], dataloaders_test, [global_model for _ in range(9)]),
                              main_title='Testing global model (before averaging) on data from different devices', color=Color.DARKCYAN)

        # Federated averaging
        federated_averaging(global_model, models)

        # Experiment 2: test the global model on the data of all devices, after it has been averaged
        multitest_classifiers(tests=zip(['Data from device ' + device for device in all_devices], dataloaders_test, [global_model for _ in range(9)]),
                              main_title='Testing global model (after averaging) on data from different devices', color=Color.CYAN)
