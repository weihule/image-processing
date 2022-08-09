import os
import numpy as np
import torch
import math


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

    def __call__(self, fpn_feature_sizes):
        """
        generate one image anchors
        :param fpn_feature_sizes: 五个特征层的宽高 [[w1, h1], [w2, h2], ...]
        :return:
        """
        one_image_anchors = list()
        # device = fpn_feature_sizes[0][0].device
        for index, _ in enumerate(self.areas):
            base_anchors = self.generate_base_anchors(self.areas[index])
            feature_anchors = self.generate_anchors_on_feature_map(base_anchors=base_anchors,
                                                                   features_map_size=fpn_feature_sizes[index],
                                                                   stride=self.strides[index])
            one_image_anchors.append(feature_anchors)

        # if input size: [640, 640]
        # [h, w, 9, 4]
        # one_image_anchors shape: [[80, 80, 9, 4], [40, 40, 9, 4], ...]
        return one_image_anchors

    def generate_base_anchors(self, area):
        # area is one of [32, 64, 128, 256, 512]
        base_anchors = np.zeros((len(self.ratios) * len(self.scales), 4), dtype=np.float32)   # [9,4]
        base_anchors[:, 2:] = area * np.tile(self.scales, (2, 3)).T         # [2, 9] -> [9, 2]
        anchor_areas = base_anchors[:, 2] * base_anchors[:, 3]
        base_anchors[:, 3] = np.sqrt(anchor_areas / np.repeat(self.ratios, 3))
        base_anchors[:, 2] = base_anchors[:, 3] * np.repeat(self.ratios, 3)

        base_anchors[:, 0] -= base_anchors[:, 2] * 0.5
        base_anchors[:, 1] -= base_anchors[:, 3] * 0.5
        base_anchors[:, 2] = base_anchors[:, 2] * 0.5
        base_anchors[:, 3] = base_anchors[:, 3] * 0.5

        # base_anchors shape: [9, 4]
        # 这里求出的是anchor相对于grid cell中心点的偏移量
        # [x_min,y_min,x_max,y_max]
        return base_anchors

    @staticmethod
    def generate_anchors_on_feature_map(base_anchors, features_map_size, stride):
        """
        生成每一个feature map上的anchors
        :param base_anchors: [9, 4]
        :param features_map_size: 五个特征层的宽高 [[w1, h1], [w2, h2], ...]
        :param stride: [8, 16, 32, 64, 128]
        :return:
        """
        # features_map_size.shape[0] is width
        # features_map_size.shape[1] is height
        shifts_x = (np.arange(0, features_map_size[0]) + 0.5) * stride  # [h, w]
        shifts_y = (np.arange(0, features_map_size[1]) + 0.5) * stride  # [h, w]

        # shifts shape: [h, w, 1, 4]
        mesh_x, mesh_y = np.meshgrid(shifts_x, shifts_y)
        # mesh_x, mesh_y = mesh_x.reshape(-1, 1), mesh_y.reshape(-1, 1)
        mesh_x, mesh_y = np.expand_dims(mesh_x, 2), np.expand_dims(mesh_y, 2)   # [h, w, 1]
        shifts = np.concatenate((mesh_x, mesh_y, mesh_x, mesh_y), axis=2)       # [h, w, 4]
        shifts = np.expand_dims(shifts, axis=2)                                 # [h, w, 1, 4]

        # base_anchors shape [9, 4] -> [1, 1, 9, 4]
        base_anchors = np.expand_dims(base_anchors, axis=0)
        base_anchors = np.expand_dims(base_anchors, axis=0)

        # [h, w, 9, 4]
        feature_map_anchors = shifts + base_anchors

        return feature_map_anchors


class RetinaAnchors1:
    def __init__(self,
                 areas=[[32, 32], [64, 64], [128, 128], [256, 256], [512,
                                                                     512]],
                 ratios=[0.5, 1, 2],
                 scales=[2**0, 2**(1.0 / 3.0), 2**(2.0 / 3.0)],
                 strides=[8, 16, 32, 64, 128]):
        self.areas = np.array(areas, dtype=np.float32)
        self.ratios = np.array(ratios, dtype=np.float32)
        self.scales = np.array(scales, dtype=np.float32)
        self.strides = np.array(strides, dtype=np.float32)

    def __call__(self, fpn_feature_sizes):
        '''
        generate one image anchors
        '''
        one_image_anchors = []
        for index, area in enumerate(self.areas):
            base_anchors = self.generate_base_anchors(area)
            feature_anchors = self.generate_anchors_on_feature_map(
                base_anchors, fpn_feature_sizes[index], self.strides[index])
            one_image_anchors.append(feature_anchors)

        # if input size:[640,640]
        # one_image_anchors shape:[[80,80,9,4],[40,40,9,4],[20,20,9,4],[10,10,9,4],[5,5,9,4]]
        # per anchor format:[x_min,y_min,x_max,y_max]
        return one_image_anchors

    def generate_base_anchors(self, area):
        '''
        generate base anchor
        '''
        # get w,h aspect ratio,shape:[9,2]
        aspects = np.array([[[s * math.sqrt(r), s * math.sqrt(1 / r)]
                             for s in self.scales] for r in self.ratios],
                           dtype=np.float32).reshape(-1, 2)
        # base anchor for each position on feature map,shape[9,4]
        base_anchors = np.zeros((len(self.scales) * len(self.ratios), 4),
                                dtype=np.float32)

        # compute aspect w\h,shape[9,2]
        base_w_h = area * aspects
        base_anchors[:, 2:] += base_w_h

        # base_anchors format: [x_min,y_min,x_max,y_max],center point:[0,0],shape[9,4]
        base_anchors[:, 0] -= base_anchors[:, 2] / 2
        base_anchors[:, 1] -= base_anchors[:, 3] / 2
        base_anchors[:, 2] /= 2
        base_anchors[:, 3] /= 2

        return base_anchors

    def generate_anchors_on_feature_map(self, base_anchors, feature_map_size,
                                        stride):
        '''
        generate one feature map anchors
        '''
        # shifts_x shape:[w],shifts_y shape:[h]
        shifts_x = (np.arange(0, feature_map_size[0]) + 0.5) * stride
        shifts_y = (np.arange(0, feature_map_size[1]) + 0.5) * stride

        # shifts shape:[w,h,2] -> [w,h,4] -> [w,h,1,4]
        shifts = np.array([[[shift_x, shift_y] for shift_y in shifts_y]
                           for shift_x in shifts_x],
                          dtype=np.float32)
        shifts = np.expand_dims(np.tile(shifts, (1, 1, 2)), axis=2)

        # base anchors shape:[9,4] -> [1,1,9,4]
        base_anchors = np.expand_dims(base_anchors, axis=0)
        base_anchors = np.expand_dims(base_anchors, axis=0)

        # generate all featrue map anchors on each feature map points
        # featrue map anchors shape:[w,h,9,4] -> [h,w,9,4]
        feature_map_anchors = np.transpose(base_anchors + shifts,
                                           axes=(1, 0, 2, 3))
        feature_map_anchors = np.ascontiguousarray(feature_map_anchors,
                                                   dtype=np.float32)

        # feature_map_anchors format: [h,w,9,4],4:[x_min,y_min,x_max,y_max]
        return feature_map_anchors


if __name__ == "__main__":
    w, h, stride = 2, 3, 6
    feature_sizes = [[80, 80], [40, 40], [20, 20], [10, 10], [5, 5]]
    ra = RetinaAnchors()
    one_image_anchors = ra(feature_sizes)

    # base_anchor = ra.generate_base_anchors(16)
    # f_m_anchor = ra.generate_anchors_on_feature_map(base_anchor, (w, h), stride)
    # f_m_anchor = np.int32(f_m_anchor)
    #
    ra1 = RetinaAnchors1()
    one_image_anchors1 = ra1(feature_sizes)
    for p, p1 in zip(one_image_anchors, one_image_anchors1):
        print(p.shape, p1.shape, (np.int32(p) == np.int32(p1)).all())
    # base_anchor1 = ra1.generate_base_anchors(16)
    # f_m_anchor1 = ra1.generate_anchors_on_feature_map(base_anchor1, (w, h), stride)
    # f_m_anchor1 = np.int32(f_m_anchor1)

    # print(base_anchor, base_anchor.shape)
    # print(base_anchor1, base_anchor1.shape)
    # print(base_anchor == base_anchor1)

    # print(f_m_anchor, f_m_anchor1.shape)
    # print(f_m_anchor1, f_m_anchor1.shape)
    # print(f_m_anchor == f_m_anchor1)


