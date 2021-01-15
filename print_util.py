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


def print_rates(tp, tn, fp, fn):
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    fpr = fp / (tn + fp)
    fnr = fn / (tp + fn)

    print(Color.BOLD + 'TPR: {:.6f} - TNR: {:.6f} - FPR: {:.6f} - FNR: {:.6f}'
          .format(tpr, tnr, fpr, fnr) + Color.END)


def print_train_classifier(epoch, num_epochs, batch, num_batches, avg_acc, lr, persistent=False, color=''):
    print('\r' + color + Color.BOLD + '[{}/{}] [{}/{}] Average acc: {:.6f} - LR: {:.6f}'
          .format(epoch+1, num_epochs, batch, num_batches, avg_acc, lr) + Color.END,
          end='\n' if persistent else '')
