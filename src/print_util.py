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


def print_positives(positives, total):
    print('Positive predictions: ' + repr(positives) + '/' + repr(total)
          + ' ({:.4f}%)'.format(100 * positives / total))


def print_rates(tp, tn, fp, fn, color=''):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)
    acc = (tp + tn) / (tp + tn + fp + fn)

    print_bar(color)
    print('TPR: {:.6f} - TNR: {:.6f} - FPR: {:.6f} - FNR: {:.6f} - ACC:{:.6f}'.format(tpr, tnr, fpr, fnr, acc))


def print_train_classifier(epoch, num_epochs, batch, num_batches, avg_acc, lr, persistent=False, color=''):
    print('\r', end='')
    print_bar(color)
    print('      [{}/{}] [{}/{}] Average acc: {:.6f} - LR: {:.6f}'
          .format(epoch + 1, num_epochs, batch, num_batches, avg_acc, lr) + Color.END,
          end='\n' if persistent else '')


def print_train_autoencoder(epoch, num_epochs, losses, lr):
    print('[{}/{}] Average loss: {:.4f}'.format(epoch + 1, num_epochs, losses.mean())
          + ' - Min loss: {:.4f}'.format(losses.min())
          + ' - 0.99 quantile: {:.4f}'.format(torch.quantile(losses, 0.99).item())
          + ' - Max loss: {:.4f}'.format(losses.max())
          + ' - STD loss: {:.4f}'.format(losses.std())
          + ' - LR: {:.6f}'.format(lr))


def print_test_autoencoder(title, losses):
    print(title + ' - Min loss: {:.4f}'.format(losses.min())
          + ' - 0.01 quantile: {:.4f}'.format(torch.quantile(losses, 0.01).item())
          + ' Average loss: {:.4f}'.format(losses.mean())
          + ' - 0.99 quantile: {:.4f}'.format(torch.quantile(losses, 0.99).item())
          + ' - Max loss: {:.4f}'.format(losses.max())
          + ' - STD loss: {:.4f}'.format(losses.std()))


def print_federation_round(federation_round, federation_rounds):
    print(Color.BOLD + '\t\t\t\t\tFEDERATION ROUND [{}/{}]\n'.format(federation_round + 1, federation_rounds) + Color.END)


def print_bar(color=''):
    if color != '':
        print(color + '| ' + Color.END, end='')
