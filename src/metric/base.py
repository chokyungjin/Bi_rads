class AverageMeter:
    """Computes and stores the average and current value

    Args:

    """

    def __init__(self):
        self.delimiter = " | "
        self.eps = 1e-6
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count + self.eps)