from __future__ import print_function, absolute_import

import numpy as np


__all__ = [
    'evaluate'
]


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
    # dismat [m, n] 含义是query中有m张图片, 每一行共n个元素, 是query (m) 中每张图片和gallery (n)中每张图片算出的距离
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print(f'Note: number of gallery samples is quite small, got {num_g}')
    # indices: [m, n]   输出按行进行排序的索引 (升序, 从小到大)
    indices = np.argsort(distmat, axis=1)
    # g_pids[indices] shape is [m, n]
    # g_pids 原来是 [n, ], g_pids[indices]操作之后, 扩展到了 [m, n]
    # 也就是每一行中的n个元素都按照 indices 中每一行的顺序进行了重排列
    # q_pids[:, np.newaxis] shape is [m, 1]
    g_pids_exp_dims, g_camids_exp_dims = g_pids[indices], g_camids[indices]
    q_pids_exp_dims = np.expand_dims(q_pids, axis=1)

    # matches中为 1 的表示query中和gallery中的行人id相同, 也就是表示同一个人
    # matches中的结果就是和后续预测出的结果进行对比的正确label
    matches = (g_pids_exp_dims == q_pids_exp_dims).astype(np.int32)      # shape is [m, n]

    # compute cmc curve for each query
    all_cmc = []
    all_ap = []
    num_valid_q = 0.    # number of valid query

    # 遍历每一张query图片
    for q_idx in range(num_q):
        # q_pid, q_camid 分别都是一个数字
        q_pid, q_camid = q_pids[q_idx], q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        # TODO: 这里要用 & ,因为前后都是np.ndarray类型, 如果前后都是list, 则可以使用 and
        removed = (g_pids_exp_dims[q_idx] == q_pid) & (g_camids_exp_dims[q_idx] == q_camid)    # [n, ]

        # keep中为True的表示gallery中符合查找条件的行人图片，
        # 这些为True的部分还需要借助matches才能完成正确的查找
        # 且keep中从左到右就是当前查找图片和每一个gallery中图片的距离距离依次递增的顺序
        keep = np.where(removed == 0, True, False)    # [n, ]

        # ===== compute cmc curve =====
        # orig_cmc中为1的位置表示查找的图片匹配正确了
        orig_cmc = matches[q_idx][keep]

        # 如果orig_cmc全为0, 也就是待查询图片没有在gallery中匹配到
        # 也就不计算top-n和ap值了
        if np.all(orig_cmc == 0):
            continue

        cmc = orig_cmc.cumsum()
        cmc = np.where(cmc >= 1, 1, 0)
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        # ===== compute average precision =====
        num_rel = orig_cmc.sum()    # 在gallery的n中图片中，匹配对了多少张
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [(x / (i + 1)) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        ap = tmp_cmc.sum() / num_rel
        all_ap.append(ap)
    
    assert num_valid_q > 0, "Error: all query identity do not appear in gallery"

    # all_cmc中一共有num_valid_q个元素, 其中每个元素又包含max_rank个值
    # 将all_cmc按列求和, 可以得到 n 个值，然后除以 num_valid_q
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(axis=0) / num_valid_q
    mAP = np.mean(all_ap)

    return all_cmc, mAP


def eval_market15012(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def evaluate(dismat, q_pids, g_pids, q_camids, g_camids, max_rank=50, use_metric_cuhk03=False):
    if use_metric_cuhk03:
        pass
    else:
        return eval_market1501(dismat, q_pids, g_pids, q_camids, g_camids, max_rank)


if __name__ == "__main__":
    arr = np.array([[1, 0, 3, 3], [2, 6, 0, 5], [1, 7, 3, 4]])  # [3, 4]
    indices = np.argsort(arr, axis=1)       # [3, 4]
    g = np.array([23, 45, 10, 4])
    q = np.array([5, 45, 7])

    print(g[indices], g[indices].shape)

    # cmcs = np.array([False, True, True, True, False, False, True])
    # cmcs = cmcs.cumsum()
    # # cmcs[cmcs > 1] = 1
    # cmcs = np.where(cmcs >= 1, 1, 0)
    # # print(cmcs)
    #
    # arrs = [[1, 2, 3], [4, 5, 6]]
    # arrs = np.asarray(arrs)
    # print(arrs.shape)

