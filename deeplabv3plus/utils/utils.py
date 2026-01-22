import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

def mkdir(path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def denormalize(data, mean, std):
    """
    归一化公式：norm = (x - mean) / std；反归一化公式：x = norm * std + mean。
    """
    if not isinstance(data, (torch.Tensor, np.ndarray)):
        raise TypeError(f"仅支持 Tensor/np.ndarray, 当前类型: {type(data)}")
    if data.ndim != 3:
        raise ValueError(f"仅支持 CHW (C, H, W) 格式，当前维度：{data.ndim}")

    if isinstance(data, torch.Tensor):
        mean = torch.tensor(mean, device=data.detach, dtype=data.dtype).view(-1, 1, 1)
        std = torch.tensor(std, device=data.detach, dtype=data.dtype).view(-1, 1, 1)
        return data * std + mean
    else:
        mean = mean.reshape(-1, 1, 1)
        std = std.reshape(-1, 1, 1)
        return data * std + mean

class Denormalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, data):
        return denormalize(data, self.mean, self.std)

def set_bn_momentum(model, momentum=0.1):
    if not 0 <= momentum <= 1:
        raise ValueError(f"momentum 必须在 0~1 之间，当前值：{momentum}")
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def freeze_bn(model):
    for m in model.modules:
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            # 可选
            for param in m.parameters():
                param.requires_grad = False
