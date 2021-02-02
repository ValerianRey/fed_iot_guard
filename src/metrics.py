import torch


class BinaryClassificationResults:
    def __init__(self, tp=0, tn=0, fp=0, fn=0):
        self.tp = tp  # Number of true positives
        self.tn = tn  # Number of true negatives
        self.fp = fp  # Number of false positives
        self.fn = fn  # Number of false negatives

    def __add__(self, other):
        results = BinaryClassificationResults(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )
        return results

    def add_tp(self, val):
        self.tp += val

    def add_tn(self, val):
        self.tn += val

    def add_fp(self, val):
        self.fp += val

    def add_fn(self, val):
        self.fn += val

    # Update the results based on the pred tensor and on the label tensor
    def update(self, pred: torch.tensor, label: torch.tensor):
        self.add_tp(torch.logical_and(torch.eq(pred, label), label.bool()).int().sum())
        self.add_tn(torch.logical_and(torch.eq(pred, label), torch.logical_not(label.bool())).int().sum())
        self.add_fp(torch.logical_and(torch.logical_not(torch.eq(pred, label)), torch.logical_not(label.bool())).int().sum())
        self.add_fn(torch.logical_and(torch.logical_not(torch.eq(pred, label)), label.bool()).int().sum())

    # True positive rate
    def tpr(self):
        return self.tp / (self.tp + self.fn)

    # True negative rate
    def tnr(self):
        return self.tn / (self.tn + self.fp)

    # False positive rate
    def fpr(self):
        return self.fp / (self.tn + self.fp)

    # False negative rate
    def fnr(self):
        return self.fn / (self.tp + self.fn)

    # Accuracy
    def acc(self):
        return (self.tp + self.tn) / self.n_samples()

    # Recall (same as true positive rate)
    def recall(self):
        return self.tpr()

    # Precision
    def precision(self):
        return self.tp / (self.tp + self.fp)

    # F1-Score
    def f1(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())

    def n_samples(self):
        return self.tp + self.tn + self.fp + self.fn

    def to_json(self):
        return {'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn}


def dumper(obj):
    try:
        return obj.to_json()
    except AttributeError:
        return obj.__dict__
