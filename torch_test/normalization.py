import torch
import torch.nn as nn
import numpy as np


def bn_process(feature, mean, var):
    feature_shape = feature.shape
    for i in range(feature_shape[1]):
        # [B, C, H, W]
        feature_t = feature[:, i, :, :]
        mean_t = feature_t.mean()  # 一个单纯的数值

        # 总体的标准差
        std_t1 = feature_t.std()
        # 样本的标准差
        std_t2 = feature_t.std(ddof=1)

        # bn_process 加上eps和pytorch保持一致
        feature[:, i, :, :] = (feature[:, i, :, :] - mean_t) / np.sqrt(std_t1 ** 2 + 1e-5)
        # update mean and var
        mean[i] = mean[i] * (1 - 0.1) + mean_t * 0.1
        var[i] = var[i] * (1 - 0.1) + (std_t2 ** 2) * 0.1

    return feature


def run_bn():
    # 随机生成一个batch为2，channel为2，height=width=2的特征向量
    feature1 = torch.randn(size=(2, 2, 3, 3))

    # 统计均值和方差
    cal_mean = [0., 0.]
    cal_var = [1.0, 1.0]

    f1 = bn_process(feature1.numpy().copy(), cal_mean, cal_var)

    bn = nn.BatchNorm2d(2, eps=1e-5)
    f2 = bn(feature1)
    print(f1)
    print(f2)


def layer_norm_process(feature, beta=0., gamma=1., eps=1e-5):
    # feature: [d1, d2, d3], var: [d1, d2], mean: [d1, d2]
    # 直接在最后一个维度做norm
    var, mean = torch.var_mean(feature, dim=-1, unbiased=False)

    # layer norm
    feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + eps)
    feature = feature * gamma + beta

    return feature


def run_ln():
    t = torch.randn(4, 2, 3)
    # 仅在最后一个维度上做norm处理
    norm = nn.LayerNorm(normalized_shape=t.shape[-1], eps=1e-5)
    t1 = norm(t)

    # 自定义norm layer
    t2 = layer_norm_process(t, eps=1e-5)

    print(t1, t1.shape)
    print(t2, t2.shape)


if __name__ == "__main__":
    # run_bn()
    run_ln()
