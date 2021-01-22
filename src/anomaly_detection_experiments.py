from anomaly_detection_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds
from architectures import SimpleAutoencoder
from data import all_devices, get_autoencoder_dataloaders
from print_util import Color, ContextPrinter


def single_autoencoder(args):
    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tSINGLE AUTOENCODER\n', bold=True)

    # Initialization of the model
    model = SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt = \
        get_autoencoder_dataloaders(args, [all_devices], ctp=ctp, color=Color.YELLOW)

    # Local training of each autoencoder
    multitrain_autoencoders(trains=zip(['with train data from all devices'], dataloaders_train, [model]),
                            lr=args.lr, epochs=args.epochs,
                            ctp=ctp, main_title='Training the single autoencoder', color=Color.GREEN)

    # Computation of the thresholds
    [threshold] = compute_thresholds(opts=zip(['with opt data from all devices'], dataloaders_opt, [model]),
                                     ctp=ctp, main_title='Computing threshold', color=Color.RED)

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
    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt =\
        get_autoencoder_dataloaders(args, all_devices, ctp=ctp, color=Color.YELLOW)

    # Local training of each autoencoder
    multitrain_autoencoders(trains=zip(['with train data from ' + device for device in all_devices], dataloaders_train, models),
                            lr=args.lr, epochs=args.epochs,
                            ctp=ctp, main_title='Training the different autoencoders', color=Color.GREEN)

    # Computation of the thresholds
    thresholds = compute_thresholds(opts=zip(['with opt data from ' + device for device in all_devices], dataloaders_opt, models),
                                    ctp=ctp, color=Color.RED)

    # Local testing of each autoencoder
    multitest_autoencoders(tests=zip(['Model trained on dataset ' + device for device in all_devices],
                                     dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt,
                                     models, thresholds),
                           ctp=ctp, main_title='Testing different clients on their own data', color=Color.BLUE)
