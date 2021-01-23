import torch
from metrics import BinaryClassificationResults


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


def print_positives(positives, total, ctp: ContextPrinter):
    ctp.print('Positive predictions: ' + repr(positives) + '/' + repr(total) + ' ({:.4f}%)'.format(100 * positives / total))


def print_rates(results: BinaryClassificationResults, ctp: ContextPrinter):
    ctp.print('TPR: {:.6f} - TNR: {:.6f} - FPR: {:.6f} - FNR: {:.6f} - ACC:{:.6f}'
              .format(results.tpr(), results.tnr(), results.fpr(), results.fnr(), results.acc()))


def print_train_classifier(epoch, num_epochs, batch, num_batches, avg_acc, lr, ctp: ContextPrinter, persistent=False):
    ctp.print('[{}/{}] [{}/{}] Average acc: {:.6f} - LR: {:.6f}'.format(epoch + 1, num_epochs, batch, num_batches, avg_acc, lr),
              rewrite=True, end='\n' if persistent else '')


def print_train_autoencoder(epoch, num_epochs, losses, lr, ctp: ContextPrinter):
    ctp.print('[{}/{}] Average loss: {:.4f}'.format(epoch + 1, num_epochs, losses.mean())
              + ' - Min loss: {:.4f}'.format(losses.min())
              + ' - 0.99 quantile: {:.4f}'.format(torch.quantile(losses, 0.99).item())
              + ' - Max loss: {:.4f}'.format(losses.max())
              + ' - STD loss: {:.4f}'.format(losses.std())
              + ' - LR: {:.6f}'.format(lr))


def print_test_autoencoder(title, losses, ctp: ContextPrinter):
    column_size = 16
    ctp.print(title.ljust(column_size)
              + '| {:.4f}'.format(losses.min()).ljust(column_size)
              + '| {:.4f}'.format(torch.quantile(losses, 0.01).item()).ljust(column_size)
              + '| {:.4f}'.format(losses.mean()).ljust(column_size)
              + '| {:.4f}'.format(torch.quantile(losses, 0.99).item()).ljust(column_size)
              + '| {:.4f}'.format(losses.max()).ljust(column_size)
              + '| {:.4f}'.format(losses.std()).ljust(column_size))


def print_loss_stats_header(ctp: ContextPrinter):
    column_size = 16
    ctp.print('LOSS STATS'.ljust(column_size)
              + '| Min'.ljust(column_size)
              + '| Q-0.01'.ljust(column_size)
              + '| Avg'.ljust(column_size)
              + '| Q-0.99'.ljust(column_size)
              + '| Max'.ljust(column_size)
              + '| Std'.ljust(column_size), bold=True)
