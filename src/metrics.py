import torch


class BinaryClassificationResult:
    def __init__(self, tp: int = 0, tn: int = 0, fp: int = 0, fn: int = 0):
        self.tp = tp  # Number of true positives
        self.tn = tn  # Number of true negatives
        self.fp = fp  # Number of false positives
        self.fn = fn  # Number of false negatives

    def __add__(self, other):
        result = BinaryClassificationResult(
            tp=self.tp + other.tp,
            tn=self.tn + other.tn,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
        )
        return result

    def add_tp(self, val: int) -> None:
        self.tp += val

    def add_tn(self, val: int) -> None:
        self.tn += val

    def add_fp(self, val: int) -> None:
        self.fp += val

    def add_fn(self, val: int) -> None:
        self.fn += val

    # Update the result based on the pred tensor and on the label tensor
    def update(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        self.add_tp(torch.logical_and(torch.eq(pred, label), label.bool()).int().sum().item())
        self.add_tn(torch.logical_and(torch.eq(pred, label), torch.logical_not(label.bool())).int().sum().item())
        self.add_fp(torch.logical_and(torch.logical_not(torch.eq(pred, label)), torch.logical_not(label.bool())).int().sum().item())
        self.add_fn(torch.logical_and(torch.logical_not(torch.eq(pred, label)), label.bool()).int().sum().item())

    # True positive rate
    def tpr(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp != 0 else 0.

    # True negative rate
    def tnr(self) -> float:
        return self.tn / (self.tn + self.fp) if self.tn != 0 else 0.

    # False positive rate
    def fpr(self) -> float:
        return self.fp / (self.tn + self.fp) if self.fp != 0 else 0.

    # False negative rate
    def fnr(self) -> float:
        return self.fn / (self.tp + self.fn) if self.fn != 0 else 0.

    # Accuracy
    def acc(self) -> float:
        return (self.tp + self.tn) / self.n_samples() if self.n_samples() != 0 else 0.

    # Balanced accuracy: equal to the accuracy as if the dataset on which we computed the results originally had benign_prop proportion of negatives
    def balanced_acc(self, benign_prop: float) -> float:
        return self.tnr() * benign_prop + self.tpr() * (1. - benign_prop)

    def __negative_minority(self) -> bool:
        return (self.tn + self.fp) < (self.n_samples() / 2)

    # Recall (same as true positive rate)
    # If minority is set to True, computes the recall of the minority class instead of the recall of the positive class
    def recall(self, minority: bool = False) -> float:
        if minority and self.__negative_minority():
            return self.tnr()
        else:
            return self.tpr()

    # Precision
    # If minority is set to True, computes the precision of the minority class instead of the precision of the positive class
    def precision(self, minority: bool = False) -> float:
        if minority and self.__negative_minority():
            return self.tn / (self.tn + self.fn) if self.tn != 0 else 0.
        else:
            return self.tp / (self.tp + self.fp) if self.tp != 0 else 0.

    # Sensitivity (same as true positive rate)
    def sensitivity(self) -> float:
        return self.tpr()

    # Specificity (same as true negative rate)
    def specificity(self) -> float:
        return self.tnr()

    # F1-Score
    # If minority is set to True, computes the F1-score of the minority class instead of the F1-score of the positive class
    def f1(self, minority: bool = False) -> float:
        return (2 * self.precision(minority) * self.recall(minority)) / (self.precision(minority) + self.recall(minority))\
            if (self.precision(minority) + self.recall(minority)) != 0 else 0.

    def n_samples(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    def to_json(self) -> dict:
        return {'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn}
