from copy import deepcopy
from types import SimpleNamespace
from typing import List, Union, Tuple, Set

import torch
import torch.nn as nn
from context_printer import Color
from context_printer import ContextPrinter as Ctp
from torch.utils.data import DataLoader

from metrics import BinaryClassificationResult
from print_util import print_train_classifier, print_train_classifier_header, print_rates


def train_classifier(model: nn.Module, params: SimpleNamespace, train_loader: DataLoader, lr_factor: float = 1.0) -> None:
    criterion = nn.BCELoss()
    optimizer = params.optimizer(model.parameters(), **params.optimizer_params)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_factor

    scheduler = params.lr_scheduler(optimizer, **params.lr_scheduler_params)

    print_train_classifier_header()
    model.train()

    for epoch in range(params.epochs):
        lr = optimizer.param_groups[0]['lr']
        result = BinaryClassificationResult()
        for i, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            pred = torch.gt(output, torch.tensor(0.5)).int()
            result.update(pred, label)

            if i % 1000 == 0:
                print_train_classifier(epoch, params.epochs, i, len(train_loader), result, lr, persistent=False)

        print_train_classifier(epoch, params.epochs, len(train_loader) - 1, len(train_loader), result, lr, persistent=True)

        scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 1e-3:
            break


def test_classifier(model: nn.Module, test_loader: DataLoader) -> BinaryClassificationResult:
    with torch.no_grad():
        model.eval()
        result = BinaryClassificationResult()
        for i, (data, label) in enumerate(test_loader):
            output = model(data)

            pred = torch.gt(output, torch.tensor(0.5)).int()
            result.update(pred, label)

        return result


# this function will train each model on its associated dataloader, and will print the title for it
# lr_factor is used to multiply the lr that is contained in params (and that should remain constant)
def multitrain_classifiers(trains: List[Tuple[str, DataLoader, nn.Module]], params: SimpleNamespace, lr_factor: float = 1.0,
                           main_title: str = 'Multitrain classifiers', color: Union[str, Color] = Color.NONE) -> None:
    Ctp.enter_section(main_title, color)
    for i, (title, dataloader, model) in enumerate(trains):
            Ctp.enter_section('[{}/{}] '.format(i + 1, len(trains)) + title, color=Color.NONE, header='      ')
            train_classifier(model, params, dataloader, lr_factor)
            Ctp.exit_section()

    Ctp.exit_section()


# this function will test each model on its associated dataloader, and will print the title for it
def multitest_classifiers(tests: List[Tuple[str, DataLoader, nn.Module]], main_title: str = 'Multitest classifiers',
                          color: Union[str, Color] = Color.NONE) -> BinaryClassificationResult:
    Ctp.enter_section(main_title, color)
    result = BinaryClassificationResult()
    for i, (title, dataloader, model) in enumerate(tests):
        Ctp.print('[{}/{}] '.format(i + 1, len(tests)) + title, bold=True)
        current_result = test_classifier(model, dataloader)
        result += current_result
        print_rates(current_result)
    Ctp.exit_section()
    Ctp.print('Average result')
    print_rates(result)
    return result
