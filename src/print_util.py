from typing import Optional

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from metrics import BinaryClassificationResult


class Columns:
    SMALL = 12
    MEDIUM = 16
    LARGE = 22


def print_federation_round(federation_round: int, n_rounds: int) -> None:
    Ctp.enter_section('Federation round [{}/{}]'.format(federation_round + 1, n_rounds), Color.DARK_GRAY)


def print_federation_epoch(epoch: int, n_epochs: int) -> None:
    Ctp.enter_section('Epoch [{}/{}]'.format(epoch + 1, n_epochs), Color.DARK_GRAY)


def print_rates(result: BinaryClassificationResult) -> None:
    Ctp.print('TPR: {:.2f}% - TNR: {:.2f}% - Accuracy: {:.2f}% - Precision: {:.2f}% - F1-Score: {:.2f}%'
              .format(result.tpr() * 100, result.tnr() * 100, result.acc() * 100, result.precision() * 100, result.f1() * 100))
    Ctp.print('TP: {} - TN: {} - FP: {} - FN:{}'.format(result.tp, result.tn, result.fp, result.fn))


def print_train_classifier_header() -> None:
    Ctp.print('Epoch'.ljust(Columns.SMALL)
              + '| Batch'.ljust(Columns.MEDIUM)
              + '| TPR'.ljust(Columns.MEDIUM)
              + '| TNR'.ljust(Columns.MEDIUM)
              + '| Accuracy'.ljust(Columns.MEDIUM)
              + '| Precision'.ljust(Columns.MEDIUM)
              + '| F1-Score'.ljust(Columns.MEDIUM)
              + '| LR'.ljust(Columns.MEDIUM), bold=True)


def print_train_classifier(epoch: int, num_epochs: int, batch: int, num_batches: int,
                           result: BinaryClassificationResult, lr: float, persistent: bool = False) -> None:
    Ctp.print('[{}/{}]'.format(epoch + 1, num_epochs).ljust(Columns.SMALL)
              + '| [{}/{}]'.format(batch + 1, num_batches).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(result.tpr()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(result.tnr()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(result.acc()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(result.precision()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(result.f1()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(lr).ljust(Columns.MEDIUM),
              rewrite=True, end='\n' if persistent else '')


def print_autoencoder_loss_header(first_column: str = 'Dataset', print_positives: bool = False, print_lr: bool = False) -> None:
    Ctp.print(first_column.ljust(Columns.MEDIUM)
              + '| Min loss'.ljust(Columns.MEDIUM)
              + '| Q-0.01 loss'.ljust(Columns.MEDIUM)
              + '| Avg loss'.ljust(Columns.MEDIUM)
              + '| Q-0.99 loss'.ljust(Columns.MEDIUM)
              + '| Max loss'.ljust(Columns.MEDIUM)
              + '| Std loss'.ljust(Columns.MEDIUM)
              + ('| Positive samples'.ljust(Columns.LARGE) + '| Positive %'.ljust(Columns.MEDIUM) if print_positives else '')
              + ('| LR'.ljust(Columns.MEDIUM) if print_lr else ''),
              bold=True)


def print_autoencoder_loss_stats(title: str, losses: torch.Tensor, positives: Optional[int] = None,
                                 n_samples: Optional[int] = None, lr: Optional[float] = None) -> None:

    print_positives = (positives is not None and n_samples is not None)
    Ctp.print(title.ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.min()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(torch.quantile(losses, 0.01).item()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.mean()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(torch.quantile(losses, 0.99).item()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.max()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.std()).ljust(Columns.MEDIUM)
              + ('| {}/{}'.format(positives, n_samples).ljust(Columns.LARGE)
                 + '| {:.4f}%'.format(100.0 * positives / n_samples).ljust(Columns.MEDIUM) if print_positives else '')
              + ('| {:.6f}'.format(lr) if lr is not None else ''))
