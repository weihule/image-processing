import os
import numpy as np
import torch
import torch.nn as nn


class RetinaAnchors:
    def __init__(self, pyramid_levels=None, strides=None, areas=None, ratios=None, scales=None):
        super(RetinaAnchors, self).__init__()

        # 如果输入是640*640, 输出的五个预测特征层的h,w是 [80, 40, 20, 10, 5]
        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            # [8, 16, 32, 64, 128]
            self.strides = [2 ** x for x in self.pyramid_levels]  # 8, 16, 32, 64, 128
        else:
            self.strides = strides

        # 即对于长宽为(原图大小/8,原图大小/8的特征图)其特征图上的每个单元格cell对应原图区域上(32,32)大小的对应区域
        if areas is None:
            # [32, 64, 128, 256, 512]
            self.areas = [2 ** (x + 2) for x in self.pyramid_levels]  # 32, 64, 128, 256, 512
        else:
            self.areas = areas

        # ratio指的是宽高比, 每种宽高比都对应三种 scale,这三种scale分别是对应的 base_size 进行放缩
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        else:
            self.scales = scales

    def __call__(self, batch_size, fpn_feature_sizes):
        """
        generate batch anchors
        :param batch_size:
        :param fpn_feature_sizes: 五个特征层的宽高 [[w1, h1], [w2, h2], ...]
        :return:
        """
        one_sample_anchors = list()
        # device = fpn_feature_sizes[0][0].device
        for index, _ in enumerate(self.areas):
            base_anchors = self.generate_base_anchors(self.areas[index])
            feature_anchors = self.generate_anchors_on_feature_map(base_anchors=base_anchors,
                                                                   features_map_size=fpn_feature_sizes[index],
                                                                   stride=self.strides[index])
            feature_anchors = feature_anchors
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

        # base_anchors shape: [9, 4] the offset of the center
        # [x_min,y_min,x_max,y_max]
        return base_anchors

    @staticmethod
    def generate_anchors_on_feature_map(base_anchors, features_map_size, stride):
        # features_map_size.shape[0] is width
        # features_map_size.shape[1] is height
        shifts_x = (np.arange(0, features_map_size[0]) + 0.5) * stride
        shifts_y = (np.arange(0, features_map_size[1]) + 0.5) * stride

        # shifts shape: [w*h, 4]
        mesh_x, mesh_y = np.meshgrid(shifts_x, shifts_y)
        mesh_y, mesh_x = mesh_y.reshape(-1, 1), mesh_x.reshape(-1, 1)
        shifts = np.concatenate((mesh_x, mesh_y, mesh_x, mesh_y), axis=1)

        # base_anchors shape [9, 4] -> [1, 1, 9, 4]
        base_anchors = np.expand_dims(base_anchors, axis=0)
        base_anchors = np.expand_dims(base_anchors, axis=0)

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
