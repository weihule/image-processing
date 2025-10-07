import numpy as np
import torch

__all__ = [
    're_ranking',
    're_ranking2',
    're_ranking3',
    'euclidean_dist'
]


def euclidean_dist(x, y):
    """

    Args:
        x: [m, d]
        y: [n, d]

    Returns:
        dist: [m, n]
    """
    m, n = x.shape[0], y.shape[0]
    xx = torch.pow(x, 2).sum(dim=-1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(dim=-1, keepdim=True).expand(n, m).T
    dist = xx + yy
    dist = torch.addmm(dist, mat1=x, mat2=y.T, beta=1, alpha=-2)
    dist = torch.clamp(dist, min=1e-12).sqrt()

    # [m, n]
    return dist


def re_ranking(prob_feature, gal_feature, k1, k2, lambda_value, local_distmat=None, only_local=False):
    """
    if use resnet50 to extract feature map
    Args:
        prob_feature: [query_num, 2048]
        gal_feature: [gallery_num, 2048]
        k1:
        k2:
        lambda_value:
        local_distmat:
        only_local:

    Returns:

    """
    # if feature vectory is numpy, you should use 'torch.tensor' transform it to tensor
    query_num = prob_feature.shape[0]
    all_num = query_num + gal_feature.shape[0]
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([prob_feature, gal_feature])  # [all_num, 2048]
        print('Using GPU to compute original distance')
        distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
                  torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).T
        distmat = torch.addmm(distmat, feat, feat.T, alpha=-2, beta=1)
        original_dist = distmat.numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    # 除以每一列的最大值并转置
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)  # [all_num, all_num] 按行

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.round(k1 / 2)) + 1]
            fi_candiadte = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candiadte]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2/3*len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)


# https://blog.csdn.net/u014453898/article/details/98790860
def re_ranking2(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    # 除以每一列的最大值并转置
    original_dist = np.transpose(1. * original_dist/np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    print('Start ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]  # 第i个图片的前20个相似图片的索引号
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]  # 返回backward_k_neigh_index中等于i的图片的行索引号
    return forward_k_neigh_index[fi]  # 返回与第i张图片 互相为k_reciprocal_neigh的图片索引号


def re_ranking3(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0)

    # 这种情况下，欧氏距离和马氏距离相同, 所以下面这行代码将余弦距离转成了欧氏距离，也就等于马氏距离
    original_dist = 2. - 2 * original_dist

    # 转换完马氏距离后，做归一化，把每列最大数max的找出来，然后让该列的全部成员除max，再转置
    original_dist = np.transpose(1. * original_dist / np.max(original_dist, axis=0, keepdims=True))  # 归一化

    # V是算出杰卡德(jaccard)距离的一个中间变量
    V = np.zeros_like(original_dist).astype(np.float32)
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1))  # 取前20，返回索引号

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)  # 取出互相是前20的
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):  # 遍历与第i张图片互相是前20的每张图片
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        # 增广k_reciprocal_neigh数据，形成k_reciprocal_expansion_index
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  # 避免重复，并从小到大排序
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])  # 第i张图片与其前20+图片的权重
        V[i, k_reciprocal_expansion_index] = 1. * weight / np.sum(
            weight)  # V记录第i个对其前20+个近邻的权重，其中有0有非0，非0表示没权重的，就似乎非前20+的

    original_dist = original_dist[:query_num, ]  # original_dist裁剪到 只有query x query+g
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(all_num):  # 遍历所有图片
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)  # 第i张图片在initial_rank前k2的序号的权重平均值
            # 第i张图的initial_rank前k2的图片对应全部图的权重平均值
            # 若V_qe中(i,j)=0，则表明i的前k2个相似图都与j不相似
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


# def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
#     # if feature vector is numpy, you should use 'torch.tensor' transform it to tensor
#     query_num = probFea.size(0)
#     all_num = query_num + galFea.size(0)
#     if only_local:
#         original_dist = local_distmat
#     else:
#         feat = torch.cat([probFea, galFea])
#         print('using GPU to compute original distance')
#         distmat = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num) + \
#                   torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
#         distmat = torch.addmm(distmat, feat, feat.T, alpha=-2, beta=1)
#         # distmat.addmm_(1,-2,feat,feat.t())
#         original_dist = distmat.numpy()
#         del feat
#         if not local_distmat is None:
#             original_dist = original_dist + local_distmat
#     gallery_num = original_dist.shape[0]
#     original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
#     V = np.zeros_like(original_dist).astype(np.float16)
#     initial_rank = np.argsort(original_dist).astype(np.int32)
#
#     print('starting re_ranking')
#     for i in range(all_num):
#         # k-reciprocal neighbors
#         forward_k_neigh_index = initial_rank[i, :k1 + 1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
#         fi = np.where(backward_k_neigh_index == i)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in range(len(k_reciprocal_index)):
#             candidate = k_reciprocal_index[j]
#             candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
#                                                :int(np.around(k1 / 2)) + 1]
#             fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
#                     candidate_k_reciprocal_index):
#                 k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
#
#         k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
#         weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
#         V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
#     original_dist = original_dist[:query_num, ]
#     if k2 != 1:
#         V_qe = np.zeros_like(V, dtype=np.float16)
#         for i in range(all_num):
#             V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
#         V = V_qe
#         del V_qe
#     del initial_rank
#     invIndex = []
#     for i in range(gallery_num):
#         invIndex.append(np.where(V[:, i] != 0)[0])
#
#     jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
#
#     for i in range(query_num):
#         temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
#         indNonZero = np.where(V[i, :] != 0)[0]
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
#                                                                                V[indImages[j], indNonZero[j]])
#         jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
#
#     final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
#     del original_dist
#     del V
#     del jaccard_dist
#     final_dist = final_dist[:query_num, query_num:]
#     return final_dist


if __name__ == "__main__":
    arr = np.array([46, 57, 23, 39, 1, 10, 0, 120])
    print(np.argpartition(arr, len(arr)-1))
