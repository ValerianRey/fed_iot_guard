from typing import Optional

import torch
from context_printer import Color
from context_printer import ContextPrinter as Ctp

from metrics import BinaryClassificationResults


class Columns:
    SMALL = 12
    MEDIUM = 16
    LARGE = 22


def print_federation_round(federation_round: int, n_rounds: int) -> None:
    Ctp.enter_section('Federation round [{}/{}]'.format(federation_round + 1, n_rounds), Color.DARK_GRAY)


def print_rates(results: BinaryClassificationResults) -> None:
    Ctp.print('TPR: {:.5f} - TNR: {:.5f} - Accuracy: {:.5f} - Recall:{:.5f} - Precision: {:.5f} - F1-Score: {:.5f}'
              .format(results.tpr(), results.tnr(), results.acc(), results.recall(), results.precision(), results.f1()))


def print_train_classifier_header() -> None:
    Ctp.print('Epoch'.ljust(Columns.SMALL)
              + '| Batch'.ljust(Columns.MEDIUM)
              + '| TPR'.ljust(Columns.MEDIUM)
              + '| TNR'.ljust(Columns.MEDIUM)
              + '| Accuracy'.ljust(Columns.MEDIUM)
              + '| Recall'.ljust(Columns.MEDIUM)
              + '| Precision'.ljust(Columns.MEDIUM)
              + '| F1-Score'.ljust(Columns.MEDIUM)
              + '| LR'.ljust(Columns.MEDIUM), bold=True)


def print_train_classifier(batch: int, num_batches: int, results: BinaryClassificationResults, lr: float, persistent: bool = False) -> None:
    Ctp.print('| [{}/{}]'.format(batch, num_batches).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(results.tpr()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(results.tnr()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(results.acc()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(results.recall()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(results.precision()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(results.f1()).ljust(Columns.MEDIUM)
              + '| {:.5f}'.format(lr).ljust(Columns.MEDIUM),
              rewrite=True, end='\n' if persistent else '')


def print_loss_autoencoder_header(first_column: str = 'Dataset', print_positives: bool = False, print_lr: bool = False) -> None:
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


def print_loss_autoencoder(title: str, losses: torch.Tensor, positives: Optional[int] = None,
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
