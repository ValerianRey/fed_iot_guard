import torch
import torch.nn as nn
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from metrics import BinaryClassificationResults
from print_util import print_loss_autoencoder, print_rates, print_loss_autoencoder_header


def train_autoencoder(model, num_epochs, train_loader, optimizer, criterion, scheduler):
    Ctp.enter_section(color=Color.BLACK)

    model.train()
    num_elements = len(train_loader.dataset)
    num_batches = len(train_loader)
    batch_size = train_loader.batch_size
    print_loss_autoencoder_header(first_column='Epoch', print_lr=True)

    for epoch in range(num_epochs):
        losses = torch.zeros(num_elements)
        for i, (x,) in enumerate(train_loader):
            output = model(x)
            # Since the normalization is made by the model itself, the output is computed on the normalized x
            # so we need to compute the loss with respect to the normalized x
            loss = criterion(output, model.normalize(x))
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            losses[start:end] = loss.mean(dim=1)

        print_loss_autoencoder('[{}/{}]'.format(epoch + 1, num_epochs), losses, lr=optimizer.param_groups[0]['lr'])
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(losses.mean())
        else:
            scheduler.step()

        if optimizer.param_groups[0]['lr'] <= 1e-3:
            break

    Ctp.exit_section()


def autoencode(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        num_elements = len(test_loader.dataset)
        num_batches = len(test_loader)
        batch_size = test_loader.batch_size

        losses = torch.zeros(num_elements)

        for i, (x,) in enumerate(test_loader):
            output = model(x)
            loss = criterion(output, model.normalize(x))

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            losses[start:end] = loss.mean(dim=1)

        return losses


def test_autoencoder(model, threshold, dataloaders, criterion) -> BinaryClassificationResults:
    Ctp.enter_section(color=Color.BLACK)
    print_loss_autoencoder_header(print_positives=True)
    results = BinaryClassificationResults()
    for key, dataloader in dataloaders.items():
        losses = autoencode(model, dataloader, criterion)
        predictions = torch.gt(losses, threshold).int()
        current_results = count_scores(predictions, is_malicious=False if key == 'benign' else True)
        title = ' '.join(key.split('_')).title()  # Transforms for example the key "mirai_ack" into the title "Mirai Ack"
        print_loss_autoencoder(title, losses, positives=current_results.tp + current_results.fp, n_samples=current_results.n_samples())
        results += current_results

    print_rates(results)
    Ctp.exit_section()
    return results


# trains should be a list of tuples (title, dataloader, model) (or a zip of the lists: titles, dataloaders, models)
# this function will train each model on its associated dataloader, and will print the title for it
def multitrain_autoencoders(trains, args, lr_factor=1.0, main_title='Multitrain autoencoders', color=Color.NONE):
    Ctp.enter_section(main_title, color)

    if type(trains) == zip:
        trains = list(trains)

    criterion = nn.MSELoss(reduction='none')
    for i, (title, dataloader, model) in enumerate(trains):
        Ctp.print('[{}/{}] '.format(i + 1, len(trains)) + title, bold=True)
        optimizer = args.optimizer(model.parameters(), **args.optimizer_params)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        scheduler = args.lr_scheduler(optimizer, **args.lr_scheduler_params)

        train_autoencoder(model, args.epochs, dataloader, optimizer, criterion, scheduler)
    Ctp.exit_section()


# opts should be a list of tuples (title, dataloader_benign_opt, model), (or a zip of the lists: titles, dataloaders_benign_opt, models)
# this function will test each model on its associated dataloader, and will find the correct threshold for them
def compute_thresholds(opts, main_title='Computing the thresholds', color=Color.NONE):
    if type(opts) == zip:
        opts = list(opts)

    Ctp.enter_section(main_title, color)

    criterion = nn.MSELoss(reduction='none')
    thresholds = []
    for i, (title, dataloader, model) in enumerate(opts):
        Ctp.print('[{}/{}] '.format(i + 1, len(opts)) + title, bold=True)
        print_loss_autoencoder_header()
        losses = autoencode(model, dataloader, criterion)
        print_loss_autoencoder('Benign (opt)', losses)
        avg_loss_val = losses.mean()
        std_loss_val = losses.std()
        threshold = avg_loss_val + std_loss_val
        thresholds.append(threshold)
        Ctp.print('The threshold is {:.4f}'.format(threshold.item()))
    Ctp.exit_section()
    return thresholds


def count_scores(predictions, is_malicious) -> BinaryClassificationResults:
    positive_predictions = predictions.sum().item()
    negative_predictions = len(predictions) - positive_predictions
    results = BinaryClassificationResults()
    if is_malicious:
        results.add_tp(positive_predictions)
        results.add_fn(negative_predictions)
    else:
        results.add_fp(positive_predictions)
        results.add_tn(negative_predictions)
    return results


# tests should be a list of tuples (title, dataloader_benign_test, dataloaders_mirai, dataloaders_gafgyt, model, threshold)
# (or a zip of the lists: titles, dataloaders, models, thresholds)
# this function will test each model on its associated dataloader, and will print the title for it
def multitest_autoencoders(tests, main_title='Multitest autoencoders', color=Color.NONE) -> BinaryClassificationResults:
    Ctp.enter_section(main_title, color)

    if type(tests) == zip:
        tests = list(tests)

    criterion = nn.MSELoss(reduction='none')
    results = BinaryClassificationResults()
    for i, (title, dataloaders, model, threshold) in enumerate(tests):
        Ctp.print('[{}/{}] '.format(i + 1, len(tests)) + title, bold=True)
        results += test_autoencoder(model, threshold, dataloaders, criterion)

    Ctp.print('Average results')
    print_rates(results)
    Ctp.exit_section()
    return results
