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
            self.strides = [2 ** x for x in self.pyramid_levels]  # 8, 16, 32, 64, 128

        # 即对于长宽为(原图大小/8,原图大小/8的特征图)其特征图上的每个单元格cell对应原图区域上(32,32)大小的对应区域
        if sizes is None:  # base_size选择范围
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]  # 32, 64, 128, 256, 512

        # ratio指的是宽高比, 每种宽高比都对应三种 scale,这三种scale分别是对应的 base_size 进行放缩
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, image):
        # 如果输入的是 640*640
        # 返回的就是 [[80,80], [40,40], [20,20], [10,10], [5,5]]
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4), dtype=np.float)

        for idx, _ in enumerate(self.pyramid_levels):
            anchors = generate_anchors(self.sizes[idx])
            shifted_anchors = shift()

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

    num_anchors = len(ratios) * len(scales)  # 9

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))  # [9, 4]

    # b = np.tile(scales, (2, len(ratios))).T
    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]  # (9, )

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, 3))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, 3)

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # 以当前anchor的中心点为坐标原点建立直角坐标系，求出左上角坐标和右下角坐标
    # anchors[:, 0::2]指的是第0列和第2列
    # anchors[:, 1::2]指的是第1列和第3列

    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    # anchors shape [9, 4]
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

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # [20, 20] [20, 20] 或 [5, 5]  [5, 5]

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
    k = shifts.shape[0]  # 对应输入的feature_map的 高*宽

    # add anchor (1, a, 4) to cell k shifts (K, 1, 4) to get (k, a, 4)
    all_anchors = (anchors.reshape((1, a, 4)) + shifts.reshape((1, k, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((a * k, 4))

    return all_anchors


class RetinaAnchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, areas=None, ratios=None, scales=None):
        super(RetinaAnchors, self).__init__()

        # 如果输入是640*640, 输出的五个预测特征层的h,w是 [80, 40, 20, 10, 5]
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]

        if strides is None:
            # [8, 16, 32, 64, 128]
            self.strides = [2 ** x for x in self.pyramid_levels]  # 8, 16, 32, 64, 128

        # 即对于长宽为(原图大小/8,原图大小/8的特征图)其特征图上的每个单元格cell对应原图区域上(32,32)大小的对应区域
        if areas is None:
            # [32, 64, 128, 256, 512]
            self.areas = [2 ** (x + 2) for x in self.pyramid_levels]  # 32, 64, 128, 256, 512

        # ratio指的是宽高比, 每种宽高比都对应三种 scale,这三种scale分别是对应的 base_size 进行放缩
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    def forward(self, batch_size, fpn_feature_sizes):
        """
        generate batch anchors
        :param batch_size:
        :param fpn_feature_sizes:
        :return:
        """
        one_sample_anchors = list()
        for index, _ in enumerate(self.areas):
            base_anchors = self.generate_base_anchors(self.areas[index])
            feature_anchors = self.generate_anchors_on_feature_map(base_anchors=base_anchors,
                                                                   features_map_size=fpn_feature_sizes[index],
                                                                   stride=self.strides[index])
            one_sample_anchors.append(feature_anchors)

        batch_anchors = list()
        # one_sample_anchors里面是一张图片的5个feature map对应的anchors
        for per_level_feature_anchors in one_sample_anchors:
            per_level_feature_anchors = torch.tile(per_level_feature_anchors,
                                                   dims=(batch_size, 1, 1))
            batch_anchors.append(per_level_feature_anchors)

        # if input size: [B, 3, 640, 640]
        # batch_anchors shape: [[B, 80*80*9, 4], [B, 40*40*9, 4], [B, 20*20*9, 4], [B, 10*10*9, 4], [B, 5*5*9, 4]]
        return batch_anchors

    def generate_base_anchors(self, area):
        base_anchors = np.zeros((len(self.ratios) * len(self.scales), 4))  # [9,4]
        base_anchors[:, 2:] = area * np.tile(self.scales, (2, 3)).T
        anchor_areas = base_anchors[:, 2] * base_anchors[:, 3]
        base_anchors[:, 2] = np.sqrt(anchor_areas / np.repeat(self.ratios, 3))
        base_anchors[:, 3] = base_anchors[:, 2] * np.repeat(self.ratios, 3)

        base_anchors[:, 0] -= base_anchors[:, 2] * 0.5
        base_anchors[:, 1] -= base_anchors[:, 3] * 0.5
        base_anchors[:, 2] = base_anchors[:, 2] * 0.5
        base_anchors[:, 3] = base_anchors[:, 3] * 0.5

        # print(base_anchors)
        # base_anchors shape: [9, 4] the offset of the center
        # [x_min,y_min,x_max,y_max]
        return torch.from_numpy(base_anchors)

    @staticmethod
    def generate_anchors_on_feature_map(base_anchors, features_map_size, stride):
        # features_map_size.shape[0] is width
        # features_map_size.shape[1] is height
        shifts_x = (torch.arange(0, features_map_size[0]) + 0.5) * stride
        shifts_y = (torch.arange(0, features_map_size[1]) + 0.5) * stride

        # this is different with numpy.meshgrid
        # shifts shape: [w*h, 4]
        mesh_y, mesh_x = torch.meshgrid(shifts_x, shifts_y)
        mesh_y, mesh_x = mesh_y.reshape(-1, 1), mesh_x.reshape(-1, 1)
        shifts = torch.cat((mesh_x, mesh_y, mesh_x, mesh_y), dim=1)

        # base_anchors shape [9, 4] -> [1, 9, 4]
        base_anchors = torch.unsqueeze(base_anchors, dim=0)

        # shifts shape [h*w, 4] -> [h*w, 1, 4]
        shifts = shifts.reshape((-1, 1, 4))

        # [h*w, 9, 4] -> [h*w*9, 4]
        feature_map_anchors = shifts + base_anchors
        feature_map_anchors = feature_map_anchors.reshape((-1, 4))

        return feature_map_anchors


if __name__ == "__main__":
    retina_anchor = RetinaAnchors()
    b_anchors = retina_anchor.generate_base_anchors(512)

    # f_size = [torch.tensor(5), torch.tensor(5)]
    # retina_anchor.generate_anchors_on_feature_map(b_anchors, f_size, 128)

    # a1 = torch.arange(8).reshape(-1, 4)
    # print(a1)
    # b1 = torch.tensor([[0, 0, 0, 0],
    #                    [1, 1, 1, 1],
    #                    [2, 2, 2, 2]])
    # a1 = torch.unsqueeze(a1, dim=0)
    # b1 = b1.reshape((-1, 1, 4))
    # res = a1 + b1

    # print(res)

    a1 = torch.arange(8).reshape((-1, 4))
    print(a1)
    a1 = torch.unsqueeze(a1, dim=0)
    batch_a1 = torch.tile(a1, (2, 1, 1))
    print(batch_a1, batch_a1.shape)
