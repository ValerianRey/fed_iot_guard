from copy import deepcopy
from types import SimpleNamespace
from typing import List, Union, Tuple, Optional

import torch
import torch.nn as nn
from context_printer import Color
from context_printer import ContextPrinter as Ctp
from torch.utils.data import DataLoader

from federated_util import s_resampling, model_canceling_attack, model_update_scaling, mimic_attack
from metrics import BinaryClassificationResult
from print_util import print_train_classifier, print_train_classifier_header, print_rates


def optimize(model: nn.Module, data: torch.Tensor, label: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module,
             result: Optional[BinaryClassificationResult] = None):
    output = model(data)
    loss = criterion(output, label)
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()

    pred = torch.gt(output, torch.tensor(0.5)).int()
    if result is not None:
        result.update(pred, label)


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
            optimize(model, data, label, optimizer, criterion, result)

            if i % 1000 == 0:
                print_train_classifier(epoch, params.epochs, i, len(train_loader), result, lr, persistent=False)

        print_train_classifier(epoch, params.epochs, len(train_loader) - 1, len(train_loader), result, lr, persistent=True)

        scheduler.step()
        if optimizer.param_groups[0]['lr'] <= 1e-3:
            break


def train_classifiers_fedsgd(global_model: nn.Module, models: List[nn.Module], dls: List[DataLoader], params: SimpleNamespace, epoch: int,
                             lr_factor: float = 1.0, mimicked_client_id: Optional[int] = None) -> None:
    criterion = nn.BCELoss()
    lr = params.optimizer_params['lr'] * lr_factor

    for model in models:
        model.train()

    result = BinaryClassificationResult()
    print_train_classifier_header()
    n_clients = len(models)

    for i, data_label_tuple in enumerate(zip(*dls)):
        for model, (data, label) in zip(models, data_label_tuple):
            optimizer = params.optimizer(model.parameters(), lr=lr, weight_decay=params.optimizer_params['weight_decay'])
            optimize(model, data, label, optimizer, criterion, result)

        malicious_clients_models = [model for client_id, model in enumerate(models) if client_id in params.malicious_clients]
        n_honest = len(models) - len(malicious_clients_models)

        # Model poisoning attacks
        if params.model_poisoning is not None:
            if params.model_poisoning == 'cancel_attack':
                model_canceling_attack(global_model=global_model, malicious_clients_models=malicious_clients_models, n_honest=n_honest)
            elif params.model_poisoning == 'mimic_attack':
                mimic_attack(models, params.malicious_clients, mimicked_client_id)
            else:
                raise ValueError('Wrong value for model_poisoning: ' + str(params.model_poisoning))

        # Rescale the model updates of the malicious clients (if any)
        model_update_scaling(global_model=global_model, malicious_clients_models=malicious_clients_models, factor=params.model_update_factor)

        if params.resampling is not None:
            models, indexes = s_resampling(models, params.resampling)
        params.aggregation_function(global_model, models)

        models = [deepcopy(global_model) for _ in range(n_clients)]

        if i % 100 == 0:
            print_train_classifier(epoch, params.epochs, i, len(dls[0]), result, lr, persistent=False)
    print_train_classifier(epoch, params.epochs, len(dls[0]) - 1, len(dls[0]), result, lr, persistent=True)


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
        Ctp.enter_section('[{}/{}] '.format(i + 1, len(trains)) + title + ' ({} samples)'.format(len(dataloader.dataset[:][0])),
                          color=Color.NONE, header='      ')
        train_classifier(model, params, dataloader, lr_factor)
        Ctp.exit_section()

    Ctp.exit_section()


# this function will test each model on its associated dataloader, and will print the title for it
def multitest_classifiers(tests: List[Tuple[str, DataLoader, nn.Module]], main_title: str = 'Multitest classifiers',
                          color: Union[str, Color] = Color.NONE) -> BinaryClassificationResult:
    Ctp.enter_section(main_title, color)
    result = BinaryClassificationResult()
    for i, (title, dataloader, model) in enumerate(tests):
        Ctp.print('[{}/{}] '.format(i + 1, len(tests)) + title + ' ({} samples)'.format(len(dataloader.dataset[:][0])), bold=True)
        current_result = test_classifier(model, dataloader)
        result += current_result
        print_rates(current_result)
    Ctp.exit_section()
    Ctp.print('Average result')
    print_rates(result)
    return result
