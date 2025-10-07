import torch
import torch.nn as nn

__all__ = [
    'CenterLoss'
]


class CenterLoss(nn.Module):
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
        if batch_size=4, feat_dim=2048, num_classes=751
        Args:
            x: [batch_size, feat_dim]
            labels: [batch_size,]
        Returns:
        """
        batch_size = x.shape[0]
        # self.centers shape is [num_classes, feat_dim]
        # distmat shape is [batch_size, num_classes]
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).T

        distmat = torch.addmm(distmat, mat1=x, mat2=self.centers.T, beta=1, alpha=-2)

        # classes shape is [batch_size, num_classes]
        classes = torch.arange(self.num_classes).long()
        classes = classes.unsqueeze(0).expand(batch_size, self.num_classes)
        if self.use_gpu:
            classes = classes.cuda()

        # labels shape is [batch_size, num_classes]
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


