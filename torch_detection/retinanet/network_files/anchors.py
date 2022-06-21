import os
import cv2
import numpy as np
import torch
import torch.nn as nn


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
    areas = anchors[:, 2] * anchors[:, 3]
    
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, 3))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, 3)
    
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    print(anchors[:, 0::2])
    print(anchors[:, 1::2])
    # anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    # anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    # print(anchors)



if __name__ == "__main__":
    # arr = np.zeros((0, 4), dtype=np.float32)
    # print(arr)

    # img_path = 'C:\\Users\\weihu\\Desktop\\images\\000.png'
    # img = cv2.imread(img_path)
    # an = Anchors()
    # res = an(img)

    generate_anchors()
    # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 1, 2, 3]])  # 3, 4
    # a_1 = np.repeat(a, repeats=2, axis=0)
    # print(a_1, a_1.shape)

    # a_2 = np.tile(a, (2, 3))
    # print(a_2, a_2.shape)

