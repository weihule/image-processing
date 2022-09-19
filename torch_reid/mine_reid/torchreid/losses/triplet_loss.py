from __future__ import print_function, absolute_import
import torch

__all__ = [
    'TripletLoss'
]


class TripletLoss:
    """
    TripletLoss with hard positive/negative mining
    """
    def __init__(self, margin=0.3):
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def __call__(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape [B, num_classes]
            targets: ground truth labels with shape [B]
        Returns:
        """
        n = inputs.shape[0]
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.T

        # beta * dist + alpha * (mat1 @ mat2)
        dist = torch.addmm(input=dist, mat1=inputs, mat2=inputs.T, beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # for each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).T)

        dist_ap, dist_an = [], []
        for i in range(n):
            # 增加一个维度方便之后用 torch.cat
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        # 使用torch.cat将list转成Tensor
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


if __name__ == "__main__":
    func = TripletLoss(margin=0.3)
    labels = torch.tensor([0, 0, 0, 2, 2, 2, 1, 1, 1])
    inputs = torch.randn(9, 5)
    func(inputs=inputs, targets=labels)

