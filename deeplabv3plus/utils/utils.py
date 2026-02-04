import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

def mkdir(path):
    if not isinstance(path, Path):
        path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


class DictWrapper:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # 递归处理嵌套字典
                setattr(self, key, DictWrapper(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        """支持字典式访问，保持兼容性"""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """支持字典式设置，保持兼容性"""
        setattr(self, key, value)

    def __repr__(self):
        """更好的调试输出"""
        return f"DictWrapper({self.__dict__})"


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

def save_ckpt(path, model, optimizer, scheduler, cur_itrs, best_score):
    """保存当前模型"""
    if not Path(path).parent.exists():
        Path(path).parent.mkdir(exist_ok=True, parents=True)

    torch.save({
        'cur_itrs': cur_itrs,
        'model_state': model.module.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_score': best_score
    }, path)

    print(f"Model Saved As {path}")


def load_checkpoint(opts, model, optimizer, scheduler, device):
    best_score = 0.0
    cur_itrs = 0

    if opts.training.ckpt is not None and Path(opts.training.ckpt).is_file():
        checkpoint = torch.load(opts.training.ckpt,
                                map_location=device)

        # 处理单GPU和多GPU模型保存的差异
        state_dict = checkpoint["model_state"]
        if not hasattr(model, 'module') and all(k.startwith('module.') for k in state_dict.items()):
            # 从 DataParallel 加载到普通模型
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif hasattr(model, 'module') and not all(k.startwith('module.') for k in state_dict.items()):
            # 从普通模型加载到 DataParallel
            state_dict = {('module.'+k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

        if opts.training.continue_training:
            if "optimizer_state" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            if "scheduler_state" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint.get("cur_itrs", 0)
            best_score = checkpoint.get("best_score", 0.0)
            print(f"Training state restored from {opts.training.ckpt}")

        print(f"Model restored from {opts.training.ckpt}")
        del checkpoint      # 释放内存

    return model, optimizer, scheduler, cur_itrs, best_score




