import os
import torch
import torch.nn.functional as F

class FocalLoss():
    def __init__(self) -> None:
        pass

if __name__ == "__main__":
    input = torch.tensor([[ 1.1041,  0.7217,  1.1316],
                [ 0.1365, -0.5008, -1.7217]])
    target = torch.tensor([0, 2])
    one_hot_code = F.one_hot(target, num_classes=3).float()
    custom_softmax = torch.exp(input) / torch.sum(torch.exp(input), dim=1).reshape(-1, 1)
    print(custom_softmax)