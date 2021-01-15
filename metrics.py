class StatisticsMeter(object):
    """Computes and stores the average, current, min and max values"""
    def __init__(self):
        self.reset()

    def reset(self):
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
