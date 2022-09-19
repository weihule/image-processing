from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CrossEntropyLabelSmooth'
]


class CrossEntropyLabelSmooth(nn.Module):
    """
    Cross Entropy loss with label smoothing regularize
    y = (1 - epsilon) * y + epsilon / K (K = num_classes)
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape [B, num_classes]
            targets: ground truth labels with shape [B]
        Returns:
        """
        log_probs = self.logsoftmax(inputs)
        # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).detach().cpu(), 1)
        targets = F.one_hot(targets, num_classes=inputs.shape[1])
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes

        # 这个就相当于 torch.sum(-targets * log_probs) / inputs.shape[0]
        loss = (-targets * log_probs).mean(0).sum()

        return loss

