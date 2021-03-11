from types import SimpleNamespace
from typing import List, Dict, Tuple, Union, Optional

import torch
import torch.nn as nn
from context_printer import Color
from context_printer import ContextPrinter as Ctp
# noinspection PyProtectedMember
from torch.utils.data import DataLoader

from architectures import Threshold
from federated_util import model_poisoning, model_aggregation
from metrics import BinaryClassificationResult
from print_util import print_autoencoder_loss_stats, print_rates, print_autoencoder_loss_header


def optimize(model: nn.Module, data: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: torch.nn.Module) -> torch.Tensor:
    output = model(data)
    # Since the normalization is made by the model itself, the output is computed on the normalized x
    # so we need to compute the loss with respect to the normalized x
    loss = criterion(output, model.normalize(data))
    optimizer.zero_grad()
    loss.mean().backward()
    optimizer.step()
    return loss


def train_autoencoder(model: nn.Module, params: SimpleNamespace, train_loader, lr_factor: float = 1.0) -> None:
    criterion = nn.MSELoss(reduction='none')
    optimizer = params.optimizer(model.parameters(), **params.optimizer_params)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_factor

    scheduler = params.lr_scheduler(optimizer, **params.lr_scheduler_params)

    model.train()
    num_elements = len(train_loader.dataset)
    num_batches = len(train_loader)
    batch_size = train_loader.batch_size
    print_autoencoder_loss_header(first_column='Epoch', print_lr=True)

    for epoch in range(params.epochs):
        losses = torch.zeros(num_elements)
        for i, (data,) in enumerate(train_loader):
            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            loss = optimize(model, data, optimizer, criterion)
            losses[start:end] = loss.mean(dim=1)

        print_autoencoder_loss_stats('[{}/{}]'.format(epoch + 1, params.epochs), losses, lr=optimizer.param_groups[0]['lr'])
        scheduler.step()


def train_autoencoders_fedsgd(global_model: nn.Module, models: List[nn.Module], dls: List[DataLoader], params: SimpleNamespace,
                              lr_factor: float = 1.0, mimicked_client_id: Optional[int] = None)\
        -> Tuple[torch.nn.Module, List[torch.nn.Module]]:
    criterion = nn.MSELoss(reduction='none')
    lr = params.optimizer_params['lr'] * lr_factor

    for model in models:
        model.train()

    for data_tuple in zip(*dls):
        for model, (data,) in zip(models, data_tuple):
            optimizer = params.optimizer(model.parameters(), lr=lr, weight_decay=params.optimizer_params['weight_decay'])
            optimize(model, data, optimizer, criterion)

        # Model poisoning attacks
        models = model_poisoning(global_model, models, params, mimicked_client_id=mimicked_client_id, verbose=False)

        # Aggregation
        global_model, models = model_aggregation(global_model, models, params, verbose=False)

    return global_model, models


def compute_reconstruction_losses(model: nn.Module, dataloader) -> torch.Tensor:
    with torch.no_grad():
        criterion = nn.MSELoss(reduction='none')
        model.eval()
        num_elements = len(dataloader.dataset)
        num_batches = len(dataloader)
        batch_size = dataloader.batch_size

        losses = torch.zeros(num_elements)

        for i, (x,) in enumerate(dataloader):
            output = model(x)
            loss = criterion(output, model.normalize(x))

            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements

            losses[start:end] = loss.mean(dim=1)

        return losses


def test_autoencoder(model: nn.Module, threshold: nn.Module, dataloaders: Dict[str, DataLoader]) -> BinaryClassificationResult:
    print_autoencoder_loss_header(print_positives=True)
    result = BinaryClassificationResult()
    for key, dataloader in dataloaders.items():
        losses = compute_reconstruction_losses(model, dataloader)
        predictions = torch.gt(losses, threshold.threshold).int()
        current_results = count_scores(predictions, is_attack=(key != 'benign'))
        title = ' '.join(key.split('_')).title()  # Transforms for example the key "mirai_ack" into the title "Mirai Ack"
        print_autoencoder_loss_stats(title, losses, positives=current_results.tp + current_results.fp, n_samples=current_results.n_samples())
        result += current_results

    return result


# this function will train each model on its associated dataloader, and will print the title for it
def multitrain_autoencoders(trains: List[Tuple[str, DataLoader, nn.Module]], params: SimpleNamespace, lr_factor: float = 1.0,
                            main_title: str = 'Multitrain autoencoders', color: Union[str, Color] = Color.NONE) -> None:
    Ctp.enter_section(main_title, color)
    for i, (title, dataloader, model) in enumerate(trains):
        Ctp.enter_section('[{}/{}] '.format(i + 1, len(trains)) + title + ' ({} samples)'.format(len(dataloader.dataset[:][0])),
                          color=Color.NONE, header='      ')
        train_autoencoder(model, params, dataloader, lr_factor)
        Ctp.exit_section()
    Ctp.exit_section()


# Compute a single threshold value. If no quantile is indicated, it's the average reconstruction loss + its standard deviation, otherwise
# it's the quantile of the loss.
def compute_threshold_value(losses: torch.Tensor, quantile: Optional[float] = None) -> torch.Tensor:
    if quantile is None:
        threshold_value = losses.mean() + losses.std()
    else:
        threshold_value = losses.quantile(quantile)

    return threshold_value


# opts should be a list of tuples (title, dataloader_benign_opt, model)
# this function will test each model on its associated dataloader, and will find the correct threshold for them
def compute_thresholds(opts: List[Tuple[str, DataLoader, nn.Module]], quantile: Optional[float] = None,
                       main_title: str = 'Computing the thresholds', color: Union[str, Color] = Color.NONE) -> List[Threshold]:

    Ctp.enter_section(main_title, color)

    thresholds = []
    for i, (title, dataloader, model) in enumerate(opts):
        Ctp.enter_section('[{}/{}] '.format(i + 1, len(opts)) + title + ' ({} samples)'.format(len(dataloader.dataset[:][0])),
                          color=Color.NONE, header='      ')
        print_autoencoder_loss_header()
        losses = compute_reconstruction_losses(model, dataloader)
        print_autoencoder_loss_stats('Benign (opt)', losses)
        threshold_value = compute_threshold_value(losses, quantile)
        threshold = Threshold(threshold_value)
        thresholds.append(threshold)
        Ctp.print('The threshold is {:.4f}'.format(threshold.threshold.item()))
        Ctp.exit_section()

    Ctp.exit_section()
    return thresholds


def count_scores(predictions: torch.Tensor, is_attack: bool) -> BinaryClassificationResult:
    positive_predictions = predictions.sum().item()
    negative_predictions = len(predictions) - positive_predictions
    results = BinaryClassificationResult()
    if is_attack:
        results.add_tp(positive_predictions)
        results.add_fn(negative_predictions)
    else:
        results.add_fp(positive_predictions)
        results.add_tn(negative_predictions)
    return results


# this function will test each model on its associated dataloader, and will print the title for it
def multitest_autoencoders(tests: List[Tuple[str, Dict[str, DataLoader], nn.Module, nn.Module]], main_title: str = 'Multitest autoencoders',
                           color: Union[str, Color] = Color.NONE) -> BinaryClassificationResult:
    Ctp.enter_section(main_title, color)

    result = BinaryClassificationResult()
    for i, (title, dataloaders, model, threshold) in enumerate(tests):
        n_samples = sum([len(dataloader.dataset[:][0]) for dataloader in dataloaders.values()])
        Ctp.enter_section('[{}/{}] '.format(i + 1, len(tests)) + title + ' ({} samples)'.format(n_samples), color=Color.NONE, header='      ')
        current_result = test_autoencoder(model, threshold, dataloaders)
        result += current_result
        Ctp.exit_section()
        print_rates(current_result)

    Ctp.exit_section()
    Ctp.print('Average result')
    print_rates(result)

    return result
