from __future__ import print_function, absolute_import
from collections import defaultdict
import torch


__all__ = [
    'AverageMeter'
]


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    avg = AverageMeter()
    print(avg.val, avg.avg, avg.sum, avg.count)
    avg.update(5, 10)
    print(avg.val, avg.avg, avg.sum, avg.count)
    avg.update(2, 4)
    print(avg.val, avg.avg, avg.sum, avg.count)


