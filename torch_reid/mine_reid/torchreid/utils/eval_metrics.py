from __future__ import print_function, absolute_import


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """
    Evaluate with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discard
    Args:
        distmat: [m, n]
        q_pids: [m, ]
        g_pids: [n, ]
        q_camids: [m, ]
        g_camids: [n, ]
        max_rank: int

    Returns:

    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g


def evaluate(dismat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        pass
    else:
        eval_market1501(dismat, q_pids, g_pids, q_camids, g_camids, max_rank)

