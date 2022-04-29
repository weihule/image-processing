import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torchvision import transforms, datasets

def test():
    input = torch.rand(2, 4)
    print(input)

if __name__ == "__main__":
    test()
