import os
import sys
import numpy as np


class YoloV3Anchors:
    def __init__(self,
                 anchor_sizes=None,
                 strides=None,
                 per_level_num_anchors=None
                 ):
        super(YoloV3Anchors, self).__init__()
        if anchor_sizes is None:
            self.anchor_sizes = [[10, 13], [16, 30], [33, 23], [30, 61],
                                 [62, 45], [59, 119], [116, 90], [156, 198],
                                 [373, 326]]
        if strides is None:
            # 输出的三个预测特征层相对原始输入的比值
            # 如果输入是416*416
            # 输出的三个特征层是[B,52,52,3,85], [B,26,26,3,85], [B,13,13,3,85]
            self.strides = [8, 16, 32]
        if per_level_num_anchors is None:
            self.per_level_num_anchors = 3

        self.anchor_sizes = np.array(self.anchor_sizes, dtype=np.float32)   # [9, 2]
        self.strides = np.array(self.strides, dtype=np.float32)
        self.per_level_anchor_sizes = self.anchor_sizes.reshape((3, 3, 2))

    def __call__(self, fpn_feature_sizes):
        """
        generate one image anchors
        """
        one_image_anchors = []
        for index, per_level_anchors in enumerate(self.per_level_anchor_sizes):
            feature_map_anchors = self.generate_anchors_on_feature_map(per_level_anchors=per_level_anchors,
                                                                       feature_map_size=fpn_feature_sizes[index],
                                                                       stride=self.strides[index])
            one_image_anchors.append(one_image_anchors)

        return one_image_anchors

    @staticmethod
    def generate_anchors_on_feature_map(per_level_anchors,
                                        feature_map_size, stride):
        """
        生成每个特征层上的anchors
        :param per_level_anchors: [3, 2] 九组预设anchor中的三个
        :param feature_map_size: 输出的预测特征图[w, h]
        :param stride: [8, 16, 32]的其中一个
        :return:
        """
        fm_h, fm_w = feature_map_size.shape[0], feature_map_size.shape[1]
        shifts_x = np.arange(0, fm_h)      # [h, w]
        shifts_y = np.arange(0, fm_w)      # [h, w]
        mesh_shifts_x, mesh_shifts_y = np.meshgrid(shifts_x, shifts_y)
        shifts = []
        for mesh_shift_x, mesh_shift_y in zip(mesh_shifts_x, mesh_shifts_y):
            mesh_shift_x, mesh_shift_y = mesh_shift_x.reshape(-1, 1), mesh_shift_y.reshape(-1, 1)
            sub_temp = np.expand_dims(np.concatenate((mesh_shift_x, mesh_shift_y), axis=-1), axis=0)    # [1, w, 2]
            shifts.append(sub_temp)
        shifts = np.concatenate(shifts, axis=0)     # [h, w, 2]
        shifts = np.expand_dims(shifts, axis=2)     # [h, w, 1, 2]
        shifts = np.tile(shifts, (1, 1, 3, 2))      # [h, w, 3, 2], 每个cell的左上角坐标重复三次

        # all_anchors_wh  [3, 2]  ->  [1, 1, 3, 2]  ->  [h, w, 3, 2]
        all_anchors_wh = np.expand_dims(np.expand_dims(per_level_anchors, axis=0), axis=0)
        all_anchors_wh = np.tile(all_anchors_wh, (fm_h, fm_w, 1, 1))

        # all_stride [1, ]  ->  [1, 1, 1, 1]  ->  [h, w, 3, 1]
        all_stride = np.expand_dims(
            np.expand_dims(
                np.expand_dims(
                    np.expand_dims(stride, axis=0), axis=0), axis=0), axis=0)
        all_stride = np.tile(all_stride, (fm_h, fm_w, 3, 1))

        # TODO: DONT FORGET
        # all_anchors_wh is relative wh on each feature map
        all_anchors_wh = all_anchors_wh / all_stride

        # feature_map_anchors: [h, w, 3, 5]
        feature_map_anchors = np.concatenate(
            (shifts, all_anchors_wh, all_stride)
        )

        return feature_map_anchors


if __name__ == "__main__":
    yolo_anchor = YoloV3Anchors()

