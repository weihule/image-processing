import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.special import logit

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class FocalLoss2(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        """
        Args:
            alpha: 类别权重，用于处理类别不平衡
            gamma: focusing参数，通常为2
        """
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        """
        Args:
            logits: (N, C) 或 (B, C, H, W) 未经softmax的logits
            target: (N, ) 类别索引，原始标签
        """
        # 适配语义分割的形状：将 (B,C,H,W) 展平为 (B*H*W, C)，target展平为 (B*H*W,)
        if len(logits.shape) == 4:
            b, c, h, w = logits.shape
            logits = logits.permute(0, 2, 3, 1).reshape(-1, c)  # [B*H*W, C]
            target = target.reshape(-1)     # [B*H*W, ]

        # 计算所有类别的预测概率 (N, C)
        p = F.softmax(logits, dim=1)
        print(p, p.shape)

        # 过滤掉ignore_index的像素
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index)
            logits = logits[valid_mask]
            target = target[valid_mask]
            p = p[valid_mask]

        # 获取真是类别概率 p_t (N, )
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)

        # 计算focal loss
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss

        return focal_loss.mean()


def test():
    focal2 = FocalLoss2(alpha=0.25, gamma=2.0)
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.randn(4, 4, 256, 256)
    target = torch.randint(0, 4, size=(4, 256, 256))

    loss = focal(logits, target)
    loss2 = focal2(logits, target)
    print(f"Focal Loss: {loss.item():.4f}  Focal Loss2: {loss2.item():.4f}")


if __name__ == "__main__":
    test()