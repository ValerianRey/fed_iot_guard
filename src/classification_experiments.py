from copy import deepcopy

from architectures import BinaryClassifier
from classification_ml import multitrain_classifiers, multitest_classifiers
from data import all_devices, get_classifier_dataloaders
from federated_util import federated_averaging
from print_util import Color, ContextPrinter


def single_classifier(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tSINGLE CLASSIFIER\n', bold=True)

    # Initialize the model
    model = BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_test = get_classifier_dataloaders(args, all_devices, ctp=ctp, color=Color.YELLOW)

    # Training
    multitrain_classifiers(trains=zip(['with data from all devices'], dataloaders_train, [model]), lr=args.lr, epochs=args.epochs,
                           ctp=ctp, main_title='Training the single model', color=Color.GREEN)

    # Testing
    multitest_classifiers(tests=zip(['Model trained on all devices'], dataloaders_test, [model]),
                          ctp=ctp, main_title='Testing the single model', color=Color.BLUE)


def multiple_classifiers(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tMULTIPLE CLASSIFIERS\n', bold=True)

    # Initialization of the models
    models = [BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers) for _ in range(len(all_devices))]

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_test = get_classifier_dataloaders(args, all_devices, ctp=ctp, color=Color.YELLOW)

    # Local training of each client
    multitrain_classifiers(trains=zip(['with data from ' + device for device in all_devices], dataloaders_train, models),
                           lr=args.lr, epochs=args.epochs,
                           ctp=ctp, main_title='Training the different clients', color=Color.GREEN)

    # Local testing
    multitest_classifiers(tests=zip(['Model trained on dataset ' + device for device in all_devices], dataloaders_test, models),
                          ctp=ctp, main_title='Testing different clients on their own data', color=Color.BLUE)


def federated_classifiers(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tFEDERATED CLASSIFIERS\n', bold=True)

    # Initialization of a global model
    global_model = BinaryClassifier(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    # Loading the data and creating the dataloaders (one per client)
    dataloaders_train, dataloaders_test = get_classifier_dataloaders(args, all_devices, ctp=ctp, color=Color.YELLOW)
    dataloaders_train = dataloaders_train[:8]  # We only see the data from 8 devices during training, so that the data from the last device is unseen

    ctp.print()

    for federation_round in range(args.federation_rounds):
        ctp.print('\t\t\t\t\tFederation round [{}/{}]'.format(federation_round + 1, args.federation_rounds), bold=True)
        ctp.add_bar(Color.BOLD)
        ctp.print()
        # Distribute the global model to all clients
        models = [deepcopy(global_model) for _ in all_devices[:8]]

        # Local training of each client
        multitrain_classifiers(trains=zip(['with data from ' + device for device in all_devices[:8]], dataloaders_train[:8], models[:8]),
                               lr=(args.lr * args.gamma_round ** federation_round), epochs=args.epochs,
                               ctp=ctp, main_title='Training the different clients', color=Color.GREEN)

        # Experiment 1: test all 8 trained clients on the data of the unseen device (9th)
        multitest_classifiers(tests=zip(['Model trained on dataset ' + device for device in all_devices[:8]],
                                        [dataloaders_test[8] for _ in range(8)],  # we always test on the same data
                                        models[:8]),
                              ctp=ctp, main_title='Testing different models on data from device ' + all_devices[8], color=Color.BLUE)

        # Experiment 2: test the global model on the data of all devices, before it has been averaged
        multitest_classifiers(tests=zip(['Data from device ' + device for device in all_devices], dataloaders_test, [global_model for _ in range(9)]),
                              ctp=ctp, main_title='Testing global model (before averaging) on data from different devices', color=Color.DARKCYAN)

        # Federated averaging
        federated_averaging(global_model, models)

        # Experiment 2: test the global model on the data of all devices, after it has been averaged
        multitest_classifiers(tests=zip(['Data from device ' + device for device in all_devices], dataloaders_test, [global_model for _ in range(9)]),
                              ctp=ctp, main_title='Testing global model (after averaging) on data from different devices', color=Color.CYAN)
        ctp.remove_header()
        ctp.print('\n')
