from __future__ import print_function, absolute_import
import torch
from .local_dist import batch_local_dist
# from local_dist import batch_local_dist

__all__ = [
    'TripletLoss',
    'hard_example_mining',
    'cal_dist_mat',
    'TripletAlignedLoss'
]


def cal_dist_mat(x1, x2):
    """
    Args:
        x1: [m, f]
        x2: [n, f]
    Returns:
        dist: [m, n]
    """
    assert len(x1.shape) == 2
    assert len(x2.shape) == 2
    assert x1.shape[1] == x2.shape[1]
    m, n = x1.shape[0], x2.shape[0]
    dist = torch.pow(x1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(x2, 2).sum(dim=1, keepdim=True).expand(n, m).T
    dist = torch.addmm(dist, mat1=x1, mat2=x2.T, beta=1, alpha=-2)
    dist = torch.clamp(dist.sqrt(), min=1e-12)

    return dist


def hard_example_mining(dist_mat, labels, return_inds=False, use_gpu=True):
    """
    for each anchor, find the hardest positive and negative sample
    Args:
        dist_mat: [N, M]
        labels: [N]
        return_inds:

    Returns:
        dist_ap: distance_ap; [N]
        dist_an: distance_an; [N]
        p_inds: LongTensor, [N]; 0 <= p_inds[i] <= N -1
        n_inds: LongTensor, [N]; 0 <= p_inds[i] <= N -1

    """
    assert len(dist_mat.shape) == 2
    assert dist_mat.shape[0] == dist_mat.shape[1]
    N = dist_mat.shape[0]

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).T)

    # dist_mat[is_pos] shape is [N, 4] 4: num_instance, 也即每个行人id有四张图片
    # dist_ap shape is [N]
    dist_ap, relative_p_ids = torch.max(dist_mat[is_pos].reshape(N, -1).contiguous(), dim=1)
    dist_an, relative_n_ids = torch.min(dist_mat[is_pos == 0].reshape(N, -1).contiguous(), dim=1)

    if return_inds:
        # shape [N, N]
        ind = torch.arange(N).expand(N, N).long()
        if use_gpu:
            ind = ind.cuda()
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].reshape(N, -1).contiguous(), dim=1, index=relative_p_ids.unsqueeze(1))
        # shape [N, 1]
        n_inds = torch.gather(
            ind[is_pos == 0].reshape(N, -1).contiguous(), dim=1, index=relative_n_ids.unsqueeze(1))
        p_inds, n_inds = p_inds.squeeze(1), n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


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
            inputs: prediction matrix (before softmax) with shape [B, feat_dim]
            targets: ground truth labels with shape [B]
            local_features: if not None, shape is [B, 128]
        Returns:
        """
        n = inputs.shape[0]
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.T

        # beta * dist + alpha * (mat1 @ mat2)
        dist = torch.addmm(input=dist, mat1=inputs, mat2=inputs.T, beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # # for each anchor, find the hardest positive and negative
        # mask = targets.expand(n, n).eq(targets.expand(n, n).T)
        #
        # dist_ap, dist_an = [], []
        # for i in range(n):
        #     # 增加一个维度方便之后用 torch.cat
        #     dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        #     dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap, dist_an = hard_example_mining(dist, targets, return_inds=False, use_gpu=True)

        # # 使用torch.cat将list转成Tensor
        # dist_ap = torch.cat(dist_ap)
        # dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss


class TripletAlignedLoss:
    """
    TripletLoss with hard positive/negative mining
    """
    def __init__(self, margin=0.3):
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def __call__(self, inputs, targets, local_features):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape [B, feat_dim]
            targets: ground truth labels with shape [B]
            local_features: shape is [B, 128]
        Returns:
        """
        print(inputs, targets, local_features)
        n = inputs.shape[0]
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.T

        # beta * dist + alpha * (mat1 @ mat2)
        dist = torch.addmm(input=dist, mat1=inputs, mat2=inputs.T, beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # [B]
        dist_ap, dist_an, p_inds, n_inds = hard_example_mining(dist, targets, return_inds=True)
        y = torch.ones_like(dist_an)

        # global loss
        global_loss = self.ranking_loss(dist_an, dist_ap, y)

        # if resnet 50, [b, 128, 8] -> [b, 8, 128]
        local_features = local_features.permute(0, 2, 1).contiguous()
        # 这样就拿到了难样本和local_features的pair_features
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)
        y = torch.ones_like(local_dist_ap)
        local_loss = self.ranking_loss(local_dist_an, local_dist_ap, y)

        return global_loss, local_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    criterion1 = TripletLoss()
    criterion2 = TripletAlignedLoss()
    inputs = torch.randn(12, 2048)
    labels = torch.tensor([3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 1])
    local_f = torch.randn(12, 128, 8)
    loss1, loss2 = criterion2(inputs=inputs,
                              targets=labels,
                              local_features=local_f)
    print(loss1, loss2)







