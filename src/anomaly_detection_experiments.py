import torch
import torch.utils.data

from anomaly_detection_ml import multitrain_autoencoders, multitest_autoencoders, compute_thresholds
from architectures import SimpleAutoencoder
from data import all_devices, mirai_attacks, gafgyt_attacks, get_autoencoder_datasets
from print_util import Color, ContextPrinter


def single_autoencoder(args):
    # Initialization of the model
    model = SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers)

    # Loading the data and creating the dataloaders
    dataset = get_autoencoder_datasets(all_devices, normalization=args.normalization)
    dataloader_train = torch.utils.data.DataLoader(dataset[0], batch_size=args.train_bs, shuffle=True)
    dataloader_opt = torch.utils.data.DataLoader(dataset[1], batch_size=args.test_bs)
    dataloader_benign_test = torch.utils.data.DataLoader(dataset[2], batch_size=args.test_bs)
    if dataset[3] is not None:
        dataloaders_mirai = [torch.utils.data.DataLoader(dataset[3][i], batch_size=args.test_bs) for i in range(len(mirai_attacks))]
    else:
        dataloaders_mirai = None
    dataloaders_gafgyt = [torch.utils.data.DataLoader(dataset[4][i], batch_size=args.test_bs) for i in range(len(gafgyt_attacks))]

    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tSINGLE AUTOENCODER\n', bold=True)

    # Local training of each autoencoder
    multitrain_autoencoders(trains=zip(['with train data from all devices'], [dataloader_train], [model]),
                            lr=args.lr, epochs=args.epochs,
                            ctp=ctp, main_title='Training the single autoencoder', color=Color.GREEN)

    # Computation of the thresholds
    [threshold] = compute_thresholds(opts=zip(['with opt data from all devices'], [dataloader_opt], [model]),
                                     ctp=ctp, main_title='Computing threshold', color=Color.RED)

    # Local testing of each autoencoder
    multitest_autoencoders(tests=zip(['Model trained on all devices'],
                                     [dataloader_benign_test], [dataloaders_mirai], [dataloaders_gafgyt],
                                     [model], [threshold]),
                           ctp=ctp, main_title='Testing the autoencoder on test data from all devices', color=Color.BLUE)


def multiple_autoencoders(args):
    # Initialization of the models
    models = [SimpleAutoencoder(activation_function=args.activation_fn, hidden_layers=args.hidden_layers) for _ in range(len(all_devices))]

    # Loading the data and creating the dataloaders
    dataloaders_train, dataloaders_opt, dataloaders_benign_test, dataloaders_mirai, dataloaders_gafgyt = [], [], [], [], []
    for device in all_devices:
        dataset = get_autoencoder_datasets([device], normalization=args.normalization)
        dataloaders_train.append(torch.utils.data.DataLoader(dataset[0], batch_size=args.train_bs, shuffle=True))
        dataloaders_opt.append(torch.utils.data.DataLoader(dataset[1], batch_size=args.test_bs))
        dataloaders_benign_test.append(torch.utils.data.DataLoader(dataset[2], batch_size=args.test_bs))
        if dataset[3] is not None:
            dataloaders_mirai.append([torch.utils.data.DataLoader(dataset[3][i], batch_size=args.test_bs) for i in range(len(mirai_attacks))])
        else:
            dataloaders_mirai.append(None)
        dataloaders_gafgyt.append([torch.utils.data.DataLoader(dataset[4][i], batch_size=args.test_bs) for i in range(len(gafgyt_attacks))])

    ctp = ContextPrinter()
    ctp.print('\n\t\t\t\t\tMULTIPLE AUTOENCODERS\n', bold=True)

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
