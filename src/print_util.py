import torch
from metrics import BinaryClassificationResults


class Columns:
    SMALL = 12
    MEDIUM = 16
    LARGE = 22


class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    BRIGHT_GREEN = '\033[92m'
    GREEN = '\033[32m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    NONE = ''


class ContextPrinter(object):
    def __init__(self):
        self.headers = []

    def add_bar(self, color):
        self.add_header(color + '█ ' + Color.END)

    def add_header(self, txt):
        self.headers.append(txt)

    def remove_header(self):
        self.headers = self.headers[:-1]

    def __print_line(self, txt='', color='', bold=False, rewrite=False, end='\n'):
        if rewrite:
            print('\r', end='')
        for header in self.headers:
            print(header, end='')

        print(color + (Color.BOLD if bold else '') + txt + Color.END, end=end)

    def print(self, txt='', color='', bold=False, rewrite=False, end='\n'):
        lines = txt.split('\n')
        for line in lines:
            self.__print_line(line, color=color, bold=bold, rewrite=rewrite, end=end)


def print_rates(results: BinaryClassificationResults, ctp: ContextPrinter):
    ctp.print('TPR: {:.6f} - TNR: {:.6f} - FPR: {:.6f} - FNR: {:.6f} - ACC:{:.6f}'
              .format(results.tpr(), results.tnr(), results.fpr(), results.fnr(), results.acc()))


def print_train_classifier_header(ctp: ContextPrinter):
    ctp.print('Epoch'.ljust(Columns.SMALL)
              + '| Batch'.ljust(Columns.MEDIUM)
              + '| Avg accuracy'.ljust(Columns.MEDIUM)
              + '| LR'.ljust(Columns.MEDIUM), bold=True)


def print_train_classifier(batch, num_batches, avg_acc, lr, ctp: ContextPrinter, persistent=False):
    ctp.print('| [{}/{}]'.format(batch, num_batches).ljust(Columns.MEDIUM)
              + '| {:.6f}'.format(avg_acc).ljust(Columns.MEDIUM)
              + '| {:.6f}'.format(lr).ljust(Columns.MEDIUM),
              rewrite=True, end='\n' if persistent else '')


def print_loss_autoencoder_header(ctp: ContextPrinter, first_column='Dataset', print_positives=False, print_lr=False):
    ctp.print(first_column.ljust(Columns.MEDIUM)
              + '| Min loss'.ljust(Columns.MEDIUM)
              + '| Q-0.01 loss'.ljust(Columns.MEDIUM)
              + '| Avg loss'.ljust(Columns.MEDIUM)
              + '| Q-0.99 loss'.ljust(Columns.MEDIUM)
              + '| Max loss'.ljust(Columns.MEDIUM)
              + '| Std loss'.ljust(Columns.MEDIUM)
              + ('| Positive samples'.ljust(Columns.LARGE) + '| Positive %'.ljust(Columns.MEDIUM) if print_positives else '')
              + ('| LR'.ljust(Columns.MEDIUM) if print_lr else ''),
              bold=True)


def print_loss_autoencoder(title, losses, ctp: ContextPrinter, positives=None, n_samples=None, lr=None):
    print_positives = (positives is not None and n_samples is not None)

    ctp.print(title.ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.min()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(torch.quantile(losses, 0.01).item()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.mean()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(torch.quantile(losses, 0.99).item()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.max()).ljust(Columns.MEDIUM)
              + '| {:.4f}'.format(losses.std()).ljust(Columns.MEDIUM)
              + ('| {}/{}'.format(positives, n_samples).ljust(Columns.LARGE)
                 + '| {:.4f}%'.format(100.0 * positives / n_samples).ljust(Columns.MEDIUM) if print_positives else '')
              + ('| {:.6f}'.format(lr) if lr is not None else ''))
