import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        # 如果输入是640*640, 输出的五个预测特征层的h,w是 [80, 40, 20, 10, 5]
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]    # 8, 16, 32, 64, 128

        # 即对于长宽为(原图大小/8,原图大小/8的特征图)其特征图上的每个单元格cell对应原图区域上(32,32)大小的对应区域
        if sizes is None:   # base_size选择范围
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]    # 32, 64, 128, 256, 512

        # ratio指的是宽高比, 每种宽高比都对应三种 scale,这三种scale分别是对应的 base_size 进行放缩
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    
    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4), dtype=np.float)

        return None


def generate_anchors(base_size=32, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # b = np.tile(scales, (2, len(ratios))).T
    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]   # (9, )
    
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, 3))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, 3)
    
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # 以当前anchor的中心点为坐标原点建立直角坐标系，求出左上角坐标和右下角坐标
    # anchors[:, 0::2]指的是第0列和第2列
    # anchors[:, 1::2]指的是第1列和第3列

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T  

    return anchors


def compute_shape(image_shape, pyramid_levels):
    # 如果输入的是 640*640
    # 返回的就是 [[80,80], [40,40], [20,20], [10,10], [5,5]]
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]

    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):
    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)  # [9, 4]
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)


def shift(shape, stride, anchors=None):
    # shape: [[80,80], [40,40], [20,20], [10,10], [5,5]]
    # stride: [8, 16, 32, 64, 128]
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)    # [20, 20] [20, 20] 或 [5, 5]  [5, 5]

    # 需要把 shifts 拼成和anchors一样的形状
    # 第一列和第三列相同, 第二列和第四列相同, 
    # 以输入shape是[10,10]为例,
    # 第一列就是网格上 10*10 个点的横坐标
    # 第一列就是网格上 10*10 个点的纵坐标
    shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
            )).transpose()

    a = anchors.shape[0]
    k = shifts.shape[0]     # 对应输入的feature_map的 高*宽

    # add anchor (1, a, 4) to cell k shifts (K, 1, 4) to get (k, a, 4)
    all_anchors = (anchors.reshape((1, a, 4)) + shifts.reshape((1, k, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((a * k, 4))

    return all_anchors


if __name__ == "__main__":
    # arr = np.zeros((0, 4), dtype=np.float32)
    # print(arr)

    # img_path = 'C:\\Users\\weihu\\Desktop\\images\\000.png'
    # img = cv2.imread(img_path)
    # an = Anchors()
    # res = an(img)

    # shift([20, 20], 32)
    # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 1, 2, 3]])  # 3, 4
    # a_1 = np.repeat(a, repeats=2, axis=0)
    # print(a_1, a_1.shape)

    # a_2 = np.tile(a, (2, 3))
    # print(a_2, a_2.shape)

    # a = np.random.random((9, 4))
    # print(a)
    # print(a[:, 0::2])
    # print(a[:, 1::2])

    # dst_list = np.arange(34)
    # batch_infer = 10
    # print(dst_list)

    # res = compute_shape([640, 640, 3], [3,4,5,6,7])
    # print(res)

    anchors = generate_anchors(base_size=256)
    shift(shape=[10, 10], stride=64, anchors=anchors)
    # K = shifts.shape[0]
    # new_shifts = shifts.reshape((1, K, 4))
    # new_shifts = np.transpose(new_shifts, (1, 0, 2))
    # print(new_shifts.shape)

