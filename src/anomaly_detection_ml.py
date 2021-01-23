import torch
import torch.nn as nn

from data import mirai_attacks, gafgyt_attacks
from print_util import print_train_autoencoder, print_test_autoencoder, Color, print_rates, print_positives, ContextPrinter, print_loss_stats_header
from metrics import BinaryClassificationResults


def train_autoencoder(model, num_epochs, train_loader, optimizer, criterion, scheduler, ctp: ContextPrinter):
    model.train()
    num_elements = len(train_loader.dataset)
    num_batches = len(train_loader)
    batch_size = train_loader.batch_size

    for epoch in range(num_epochs):
        losses = torch.zeros(num_elements)
        for i, (x,) in enumerate(train_loader):
            output = model(x)
            loss = criterion(output, x)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            losses[start:end] = loss.mean(dim=1)

        print_train_autoencoder(epoch, num_epochs, losses, optimizer.param_groups[0]['lr'], ctp)
        scheduler.step(losses.mean())
        if optimizer.param_groups[0]['lr'] <= 1e-3:
            break


def autoencode(model, test_loader, criterion, ctp: ContextPrinter, title=''):
    with torch.no_grad():
        model.eval()
        num_elements = len(test_loader.dataset)
        num_batches = len(test_loader)
        batch_size = test_loader.batch_size

        losses = torch.zeros(num_elements)

        for i, (x,) in enumerate(test_loader):
            output = model(x)
            loss = criterion(output, x)

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            losses[start:end] = loss.mean(dim=1)

        print_test_autoencoder(title, losses, ctp)
        return losses


def test_autoencoder(model, threshold, dataloader_benign_test, dataloaders_gafgyt, dataloaders_mirai, criterion, ctp):
    ctp.add_bar(Color.NONE)
    print_loss_stats_header(ctp)
    losses = autoencode(model, dataloader_benign_test, criterion, ctp, '[Benign (test)]')
    predictions = torch.gt(losses, threshold).int()
    results = count_scores(predictions, is_malicious=False, ctp=ctp)
    # Mirai validation
    if dataloaders_mirai is not None:
        for j, attack in enumerate(mirai_attacks):
            losses = autoencode(model, dataloaders_mirai[j], criterion, ctp, '[Mirai ' + attack + ']')
            predictions = torch.gt(losses, threshold)
            results += count_scores(predictions, is_malicious=True, ctp=ctp)
    # Gafgyt validation
    for j, attack in enumerate(gafgyt_attacks):
        losses = autoencode(model, dataloaders_gafgyt[j], criterion, ctp, '[Gafgyt ' + attack + ']')
        predictions = torch.gt(losses, threshold)
        results += count_scores(predictions, is_malicious=True, ctp=ctp)
    print_rates(results, ctp)
    ctp.remove_header()
    return results


# trains should be a list of tuples (title, dataloader, model) (or a zip of the lists: titles, dataloaders, models)
# this function will train each model on its associated dataloader, and will print the title for it
def multitrain_autoencoders(trains, args, ctp: ContextPrinter, lr_factor=1.0, main_title='Multitrain autoencoders', color=Color.NONE):
    ctp.print(main_title, color=color, bold=True)
    ctp.add_bar(color)

    if type(trains) == zip:
        trains = list(trains)

    criterion = nn.MSELoss(reduction='none')
    for i, (title, dataloader, model) in enumerate(trains):
        ctp.print('[{}/{}] '.format(i + 1, len(trains)) + title)
        optimizer = args.optimizer(model.parameters(), **args.optimizer_params)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        scheduler = args.lr_scheduler(optimizer, **args.lr_scheduler_params)

        train_autoencoder(model, args.epochs, dataloader, optimizer, criterion, scheduler, ctp)
        if i != len(trains) - 1:
            ctp.print()
    ctp.remove_header()


# opts should be a list of tuples (title, dataloader_benign_opt, model), (or a zip of the lists: titles, dataloaders_benign_opt, models)
# this function will test each model on its associated dataloader, and will find the correct threshold for them
def compute_thresholds(opts, ctp: ContextPrinter, main_title='Computing thresholds', color=Color.NONE):
    if type(opts) == zip:
        opts = list(opts)

    ctp.print(main_title, color=color, bold=True)
    ctp.add_bar(color)
    criterion = nn.MSELoss(reduction='none')
    thresholds = []
    for i, (title, dataloader, model) in enumerate(opts):
        ctp.print('[{}/{}] '.format(i + 1, len(opts)) + title)
        print_loss_stats_header(ctp)
        losses = autoencode(model, dataloader, criterion, ctp, '[Benign (opt)]')
        avg_loss_val = losses.mean()
        std_loss_val = losses.std()
        threshold = avg_loss_val + std_loss_val
        thresholds.append(threshold)
        ctp.print('The threshold is {:.4f}'.format(threshold.item()))
        if i != len(opts) - 1:
            ctp.print()
    ctp.remove_header()
    return thresholds


def count_scores(predictions, is_malicious, ctp: ContextPrinter):
    positive_predictions = predictions.sum().item()
    negative_predictions = len(predictions) - positive_predictions
    print_positives(positive_predictions, len(predictions), ctp)
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
def multitest_autoencoders(tests, ctp: ContextPrinter, main_title='Multitest autoencoders', color=Color.NONE):
    ctp.print(main_title, color=color, bold=True)
    ctp.add_bar(color)

    if type(tests) == zip:
        tests = list(tests)

    criterion = nn.MSELoss(reduction='none')
    results = BinaryClassificationResults()
    for i, (title, dataloader_benign_test, dataloaders_mirai, dataloaders_gafgyt, model, threshold) in enumerate(tests):
        ctp.print('[{}/{}] '.format(i + 1, len(tests)) + title, bold=True)
        results += test_autoencoder(model, threshold, dataloader_benign_test, dataloaders_gafgyt, dataloaders_mirai, criterion, ctp)
        ctp.print()

    ctp.print('Average results')
    print_rates(results, ctp)
    ctp.remove_header()
