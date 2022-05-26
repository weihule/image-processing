import torch
import torch.nn as nn
from collections import OrderedDict
from functools import partial

def add_func(a, b, c):
    return a + b + c

if __name__ == "__main__":
    add_test1 = partial(add_func, 1, 3)
    print(add_test1(2))
    print(add_test1(4))