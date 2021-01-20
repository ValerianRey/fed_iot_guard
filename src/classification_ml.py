from metrics import StatisticsMeter
import torch
from print_util import print_train_classifier, Color, print_bar, print_rates
import torch.nn as nn


def train_classifier(model, num_epochs, train_loader, optimizer, criterion, scheduler, color=''):
    model.train()

    num_elements = len(train_loader.dataset)
    num_batches = len(train_loader)
    batch_size = train_loader.batch_size

    for epoch in range(num_epochs):
        accuracy = StatisticsMeter()
        lr = optimizer.param_groups[0]['lr']

        for i, (data, label) in enumerate(train_loader):
            output = model(data)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            predictions = torch.gt(output, torch.tensor(0.5)).int()
            success = torch.eq(predictions, label).float()

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            accuracy.update(success.mean(), end-start)
            if i % 1000 == 0:
                print_train_classifier(epoch, num_epochs, i, len(train_loader), accuracy.avg, lr, persistent=False, color=color)

        print_train_classifier(epoch, num_epochs, len(train_loader), len(train_loader), accuracy.avg, lr, persistent=True, color=color)

        scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 1e-3:
            break


def test_classifier(model, test_loader):
    with torch.no_grad():
        model.eval()

        num_elements = len(test_loader.dataset)
        num_batches = len(test_loader)
        batch_size = test_loader.batch_size

        predictions = torch.zeros(num_elements)
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i, (data, label) in enumerate(test_loader):
            output = model(data)

            pred = torch.gt(output, torch.tensor(0.5)).int()
            tp += torch.logical_and(torch.eq(pred, label), label.bool()).int().sum()
            tn += torch.logical_and(torch.eq(pred, label), torch.logical_not(label.bool())).int().sum()
            fp += torch.logical_and(torch.logical_not(torch.eq(pred, label)), torch.logical_not(label.bool())).int().sum()
            fn += torch.logical_and(torch.logical_not(torch.eq(pred, label)), label.bool()).int().sum()

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            predictions[start:end] = pred.squeeze()

        return tp, tn, fp, fn


# tests should be a list of tuples (title, dataloader, model) (or a zip of the lists: titles, dataloaders, models)
# this function will test each model on its associated dataloader, and will print the title for it
def multitest_classifiers(tests, main_title='Multitest classifiers', color=''):
    print(Color.BOLD + color + main_title + Color.END)
    if type(tests) == zip:
        tests = list(tests)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i, (title, dataloader, model) in enumerate(tests):
        print_bar(color)
        print('[{}/{}] '.format(i + 1, len(tests)) + title)
        current_tp, current_tn, current_fp, current_fn = test_classifier(model, dataloader)
        print_rates(current_tp, current_tn, current_fp, current_fn, color)
        tp += current_tp
        tn += current_tn
        fp += current_fp
        fn += current_fn
        print_bar(color)
        print()

    print_bar(color)
    print('Average results')
    print_rates(tp, tn, fp, fn, color)
    print('\n')


def multitrain_classifiers(trains, lr, epochs, main_title='Multitrain autoencoders', color=''):
    print(Color.BOLD + color + main_title + Color.END)
    if type(trains) == zip:
        trains = list(trains)

    criterion = nn.BCELoss()
    for i, (title, dataloader, model) in enumerate(trains):
        print_bar(color)
        print('[{}/{}] '.format(i + 1, len(trains)) + title)

        optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        train_classifier(model, epochs, dataloader, optimizer, criterion, scheduler, color)
        if i != len(trains)-1:
            print_bar(color)
        print()
    print()
