import torch
from print_util import print_train_autoencoder, print_test_autoencoder


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

        print_train_autoencoder(epoch, num_epochs, losses, optimizer.param_groups[0]['lr'])
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

        print_test_autoencoder(title, losses)

        return losses
