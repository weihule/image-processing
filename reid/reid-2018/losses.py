from __future__ import absolute_import
import sys

import torch
from torch import nn

"""
Shorthands for loss:
- CrossEntropyLabelSmooth: xent
- TripletLoss: htri
- CenterLoss: cent
"""
__all__ = ['DeepSupervision',
           'CrossEntropyLabelSmooth',
           'TripletLoss',
           'CenterLoss',
           'RingLoss']


def DeepSupervision(criterion, xs, y):
    """
    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    return loss


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
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin

        # ranking_loss主要算的就是 ReLU(ap - y*an + margin)
        # 这里y取1, 并用ReLU()激活
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size)
        """
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(mat1=inputs, mat2=inputs.T, beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # beta(dist) + alpha(a @ b)
        # dist = torch.addmm(dist, mat1=inputs, mat2=inputs.t(), beta=1, alpha=-2)
        # dist = torch.sqrt(torch.clamp(dist, min=1e-12))    # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_ap = torch.cat(dist_ap)    # 这一步其实就相当于是将list转成tensor, shape is [batch_size, ]
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)

        # TODO: dist_an, dist_ap 顺序不要写反了
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            if batch_size = 4, feat_dim=2048, num_classes=751
            x: feature matrix with shape (batch_size, feat_dim). (4, 2048)
            labels: ground truth labels with shape (batch_size). (4)
        """
        batch_size = x.size(0)
        # self.centers shape is [751, 2048]
        # distmat shape is [4, 751]
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()

        # distmat = distmat + (-2)*([4, 2048] @ [2048, 751])
        distmat.addmm_(mat1=x, mat2=self.centers.t(), beta=1, alpha=-2)

        # classes [751] -> [1, 751] -> [4, 751]
        classes = torch.arange(self.num_classes).long()
        classes = classes.unsqueeze(0).expand(batch_size, self.num_classes)
        if self.use_gpu:
            classes = classes.cuda()

        # labels [4] -> [4, 1] -> [4, 751]
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        # mask = labels.eq(classes.expand(batch_size, self.num_classes))
        mask = labels.eq(classes)

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)   # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss


class RingLoss(nn.Module):
    """Ring loss.
    
    Reference:
    Zheng et al. Ring loss: Convex Feature Normalization for Face Recognition. CVPR 2018.
    """
    def __init__(self, weight_ring=1.):
        super(RingLoss, self).__init__()
        self.radius = nn.Parameter(torch.ones(1, dtype=torch.float))
        self.weight_ring = weight_ring

    def forward(self, x):
        l = ((x.norm(p=2, dim=1) - self.radius)**2).mean()
        return l * self.weight_ring


if __name__ == '__main__':
    # arr = torch.tensor([[1, 0, 2, 1, 3], [0, 1, 2, 3, 0],
    #                     [1, 2, 3, 1, 1], [1, 0, 0, 1, 0]])
    # a_2 = torch.pow(arr, 2).sum(dim=1, keepdim=True).expand(4, 4)
    # b_2 = a_2.T
    # a_b = arr @ arr.T
    # dist = (a_2 + b_2 - 2*a_b).sqrt()

    ins = torch.randn(8, 10)
    tars = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2]).long()
    trip = TripletLoss()
    res = trip(ins, tars)

