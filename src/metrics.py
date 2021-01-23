class StatisticsMeter(object):
    """Computes and stores the average, current, min and max values"""
    def __init__(self):
        self.current_value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.min = None
        self.max = None

    def update(self, val, n=1):
        self.current_value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.min is None:
            self.min = val
        else:
            self.min = min(self.min, val)
        if self.max is None:
            self.max = val
        else:
            self.max = max(self.max, val)


class BinaryClassificationResults(object):
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
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

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
