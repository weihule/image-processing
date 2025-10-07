import torch


__all__ = [
    'batch_euclidean_dist',
    'shortest_dist',
    'batch_local_dist'
]


def batch_euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [batch_size, local_part1, feature_channel]
        y: pytorch Variable, with shape [batch_size, local_part2, feature_channel]

    Returns:
        dist: pytorch Variable, with shape [batch_size, local_part1, local_part2]
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[-1] == y.shape[-1]

    N, m, d = x.shape
    N, n, d = y.shape

    xx = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(N, m, n)
    yy = torch.pow(y, 2).sum(dim=-1, keepdim=True).expand(N, n, m).permute(0, 2, 1).contiguous()
    dist = xx + yy
    dist = torch.baddbmm(dist, batch1=x, batch2=y.permute(0, 2, 1).contiguous(), beta=1, alpha=-2)
    dist = torch.clamp(dist, min=1e-12).sqrt()

    return dist


def shortest_dist(dist_mat):
    """
    Args:
        dist_mat: availabel shape:
        1): [m, n]
        2): [N, m, n], N is batch_size
        3): [*, m, n], * can be arbitrary additional dimensions

    Returns:
        1): scalar
        2): [N, ]
        3): *
    """
    # 如果dist_mat是二维，就在最前面增加一个维度，让其变成三维
    if dist_mat.shape == 2:
        dist_mat = dist_mat.unsqueeze(0)
    # m, n, N = dist_mat.shape
    N, m, n = dist_mat.shape
    res = [0 for _ in range(N)]
    for N_index in range(N):
        cur_dist_mat = dist_mat[N_index]    # [m, n]
        dist = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if (i == 0) and (j == 0):
                    dist[i][j] = cur_dist_mat[i][j]
                elif (i == 0) and (j > 0):
                    dist[i][j] = dist[i][j-1]+cur_dist_mat[i][j]
                elif (i > 0) and (j == 0):
                    dist[i][j] = dist[i-1][j]+cur_dist_mat[i][j]
                else:
                    dist[i][j] = min(dist[i][j-1], dist[i-1][j]) + cur_dist_mat[i][j]

        res[N_index] = dist[-1][-1]

    return torch.tensor(res, dtype=torch.float32)


def batch_local_dist(x, y):
    """
    Args:
        x: [N, m, d]
        y: [N, n, d]

    Returns:
        dist: [N, ]
    """
    assert len(x.shape) == 3
    assert len(y.shape) == 3
    assert x.shape[0] == y.shape[0]
    assert x.shape[-1] == y.shape[-1]

    # shape is [N, m, n]
    dist_mat = batch_euclidean_dist(x, y)
    # TODO: 归一化, 为了训练的稳定性
    dist_mat = (torch.exp(dist_mat) - 1.) / (torch.exp(dist_mat) + 1.)
    # shape [N, ]
    # dist = shortest_dist(dist_mat.permute(1, 2, 0).contiguous())
    dist = shortest_dist(dist_mat)

    return dist


if __name__ == "__main__":
    x = torch.randn(4, 8, 128)
    y = torch.randn(4, 8, 128)
    dist_map = batch_local_dist(x, y)
    print(dist_map, dist_map.shape)






