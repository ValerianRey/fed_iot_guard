from copy import deepcopy

from anomaly_detection_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds
from architectures import SimpleAutoencoder
from data import all_devices, get_autoencoder_dataloaders
from federated_util import federated_averaging
from print_util import Color, ContextPrinter


def single_autoencoder(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tSINGLE AUTOENCODER\n', bold=True)

    # Initialization of the model
    model = SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt = \
        get_autoencoder_dataloaders(args, [all_devices], ctp=ctp, color=Color.YELLOW)
    ctp.print('\n')

    # Local training of each autoencoder
    multitrain_autoencoders(trains=zip(['with train data from all devices'], dataloaders_train, [model]),
                            args=args, ctp=ctp, main_title='Training the single autoencoder', color=Color.GREEN)
    ctp.print('\n')

    # Computation of the thresholds
    [threshold] = compute_thresholds(opts=zip(['with opt data from all devices'], dataloaders_opt, [model]),
                                     ctp=ctp, main_title='Computing threshold', color=Color.RED)
    ctp.print('\n')

    # Local testing of each autoencoder
    multitest_autoencoders(tests=zip(['Model trained on all devices'],
                                     dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt,
                                     [model], [threshold]),
                           ctp=ctp, main_title='Testing the autoencoder on test data from all devices', color=Color.BLUE)


def multiple_autoencoders(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tMULTIPLE AUTOENCODERS\n', bold=True)

    # Initialization of the models
    models = [SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers) for _ in range(len(all_devices))]

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt = \
        get_autoencoder_dataloaders(args, all_devices, ctp=ctp, color=Color.YELLOW)
    ctp.print('\n')

    # Local training of each autoencoder
    multitrain_autoencoders(trains=zip(['with train data from ' + device for device in all_devices], dataloaders_train, models),
                            args=args, ctp=ctp, main_title='Training the different autoencoders', color=Color.GREEN)
    ctp.print('\n')

    # Computation of the thresholds
    thresholds = compute_thresholds(opts=zip(['with opt data from ' + device for device in all_devices], dataloaders_opt, models),
                                    ctp=ctp, color=Color.RED)
    ctp.print('\n')

    # Local testing of each autoencoder
    multitest_autoencoders(tests=zip(['Model trained on dataset ' + device for device in all_devices],
                                     dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt,
                                     models, thresholds),
                           ctp=ctp, main_title='Testing different clients on their own data', color=Color.BLUE)


def federated_autoencoders(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tFEDERATED AUTOENCODERS\n', bold=True)

    # Initialization of a global model
    global_model = SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)
    global_threshold = 1.0  # Arbitrary initialization of the threshold of the global model

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt = \
        get_autoencoder_dataloaders(args, all_devices, ctp=ctp, color=Color.YELLOW)
    dataloaders_train = dataloaders_train[:8]  # We only see the data from 8 devices during training, so that the data from the last device is unseen
    dataloaders_opt = dataloaders_opt[:8]
    ctp.print('\n')

    for federation_round in range(args.federation_rounds):
        ctp.print('\t\t\t\t\tFederation round [{}/{}]'.format(federation_round + 1, args.federation_rounds), bold=True)
        ctp.add_bar(Color.BOLD)
        ctp.print()

        # Distribute the global model to all clients
        models = [deepcopy(global_model) for _ in all_devices[:8]]

        # Local training of each client
        multitrain_autoencoders(trains=zip(['with data from ' + device for device in all_devices[:8]], dataloaders_train, models),
                                args=args, ctp=ctp, lr_factor=(args.gamma_round ** federation_round),
                                main_title='Training the different clients', color=Color.GREEN)
        ctp.print('\n')

        # Computation of the thresholds
        thresholds = compute_thresholds(opts=zip(['with opt data from ' + device for device in all_devices[:8]], dataloaders_opt, models),
                                        ctp=ctp, color=Color.RED)
        ctp.print('\n')

        # Experiment 1: test all 8 trained clients on the data of the unseen device (9th)
        multitest_autoencoders(tests=zip(['Model trained on dataset ' + device for device in all_devices],
                                         [dataloaders_benign_test[8] for _ in range(8)],
                                         [dataloaders_mirai[8] for _ in range(8)],
                                         [dataloaders_gafgyt[8] for _ in range(8)],
                                         models, thresholds),
                               ctp=ctp, main_title='Testing different clients on their own data', color=Color.BLUE)
        ctp.print('\n')

        # Experiment 2: test the global model on the data of all devices, before it has been averaged
        multitest_autoencoders(tests=zip(['Data from device ' + device for device in all_devices],
                                         dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt,
                                         [global_model for _ in range(9)],
                                         [global_threshold for _ in range(9)]),
                               ctp=ctp, main_title='Testing global model (before averaging) on data from different devices', color=Color.DARKCYAN)
        ctp.print('\n')

        # Federated averaging
        federated_averaging(global_model, models)
        global_threshold = sum(thresholds) / len(thresholds)

        # Experiment 3: test the global model on the data of all devices, after it has been averaged
        multitest_autoencoders(tests=zip(['Data from device ' + device for device in all_devices],
                                         dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt,
                                         [global_model for _ in range(9)],
                                         [global_threshold for _ in range(9)]),
                               ctp=ctp, main_title='Testing global model (after averaging) on data from different devices', color=Color.CYAN)
        ctp.remove_header()
        ctp.print('\n')
