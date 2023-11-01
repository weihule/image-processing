from __future__ import print_function, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'CrossEntropyLabelSmooth'
]


# class CrossEntropyLabelSmooth(nn.Module):
#     """
#     Cross Entropy loss with label smoothing regularize
#     y = (1 - epsilon) * y + epsilon / K (K = num_classes)
#     """
#     def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
#         super(CrossEntropyLabelSmooth, self).__init__()
#         self.num_classes = num_classes
#         self.epsilon = epsilon
#         self.use_gpu = use_gpu
#         self.logsoftmax = nn.LogSoftmax(dim=1)
#
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: prediction matrix (before softmax) with shape [B, num_classes]
#             targets: ground truth labels with shape [B]
#         Returns:
#         """
#         log_probs = self.logsoftmax(inputs)
#         # targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).detach().cpu(), 1)
#         targets = F.one_hot(targets, num_classes=inputs.shape[1])
#         if self.use_gpu:
#             targets = targets.cuda()
#         targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
#
#         # 这个就相当于 torch.sum(-targets * log_probs) / inputs.shape[0]
#         loss = (-targets * log_probs).mean(0).sum()
#
#         return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
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
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


import math 
import numpy as np
def test():
    x = np.array([1.0, 5.0, 4.0])
    label = np.array([0., 1.0, 0.0])
    y_hat = np.array([0.013, 0.721, 0.265])
    eplise = 0.001
    new_label = label * (1-0.001) + (0.001 / 3)
    print(new_label)

    print(-label*np.log(y_hat))
    print(-new_label*np.log(y_hat))

if __name__ == "__main__":
    test()

