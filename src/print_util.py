import torch


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
        self.add_header(color + '| ' + Color.END)

    def add_header(self, txt):
        self.headers.append(txt)

    def remove_header(self):
        self.headers = self.headers[:-1]

    def print(self, txt='', color='', bold=False, rewrite=False, end='\n'):
        if rewrite:
            print('\r', end='')
        for header in self.headers:
            print(header, end='')

        print(color + (Color.BOLD if bold else '') + txt + Color.END, end=end)


def print_positives(positives, total, ctp: ContextPrinter):
    ctp.print('Positive predictions: ' + repr(positives) + '/' + repr(total) + ' ({:.4f}%)'.format(100 * positives / total))


def print_rates(tp, tn, fp, fn, ctp: ContextPrinter):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)

    ctp.print('TPR: {:.6f} - TNR: {:.6f} - FPR: {:.6f} - FNR: {:.6f} - ACC:{:.6f}'.format(tpr, tnr, fpr, fnr, acc))


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
    ctp.print(title + ' - Min loss: {:.4f}'.format(losses.min())
              + ' - 0.01 quantile: {:.4f}'.format(torch.quantile(losses, 0.01).item())
              + ' Average loss: {:.4f}'.format(losses.mean())
              + ' - 0.99 quantile: {:.4f}'.format(torch.quantile(losses, 0.99).item())
              + ' - Max loss: {:.4f}'.format(losses.max())
              + ' - STD loss: {:.4f}'.format(losses.std()))
