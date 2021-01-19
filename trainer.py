import torch
from metrics import StatisticsMeter
from print_util import Color, print_train_classifier, print_rates, print_bar


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

    return model


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


# experiments should be a list of tuples (title, dataloader, model) (or a zip of the lists: titles, dataloaders, models)
# this function will test each model on its associated dataloader, and will print the title for it
def multitest_classifiers(experiments, main_title='Multitest classifiers', color=''):
    print(Color.BOLD + color + main_title + Color.END)
    if type(experiments) == zip:
        experiments = list(experiments)

    tp, tn, fp, fn = 0, 0, 0, 0
    for i, (title, dataloader, model) in enumerate(experiments):
        print_bar(color)
        print('[{}/{}] '.format(i + 1, len(experiments)) + title)
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
    print()
    return tp, tn, fp, fn


def train_autoencoder(model, num_epochs, train_loader, optimizer, criterion, scheduler):
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

        print('[{}/{}] Average loss: {:.4f}'.format(epoch + 1, num_epochs, losses.mean())
              + ' - Min loss: {:.4f}'.format(losses.min())
              + ' - 0.99 quantile: {:.4f}'.format(torch.quantile(losses, 0.99).item())
              + ' - Max loss: {:.4f}'.format(losses.max())
              + ' - STD loss: {:.4f}'.format(losses.std())
              + ' - LR: {:.6f}'.format(optimizer.param_groups[0]['lr']))

        scheduler.step(losses.mean())
        if optimizer.param_groups[0]['lr'] <= 1e-3:
            break


def test_autoencoder(model, test_loader, criterion, title=''):
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

        print(title + ' - Min loss: {:.4f}'.format(losses.min())
              + ' - 0.01 quantile: {:.4f}'.format(torch.quantile(losses, 0.01).item())
              + ' Average loss: {:.4f}'.format(losses.mean())
              + ' - 0.99 quantile: {:.4f}'.format(torch.quantile(losses, 0.99).item())
              + ' - Max loss: {:.4f}'.format(losses.max())
              + ' - STD loss: {:.4f}'.format(losses.std()))

        return losses
