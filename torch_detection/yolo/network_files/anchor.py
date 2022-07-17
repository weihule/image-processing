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

        self.anchor_sizes = np.array(self.anchor_sizes, dtype=np.float32)
        self.strides = np.array(self.strides, dtype=np.float32)
        self.per_level_anchor_sizes = self.anchor_sizes.reshape((3, 3, 2))

    def __call__(self, fpn_feature_sizes):
        """
        generate one image anchors
        """
        one_image_anchors = []
        for index, per_level_anchors in enumerate(self.per_level_anchor_sizes):
            pass

    def generate_anchors_on_feature_map(self, per_level_anchors,
                                        feature_map_size, stride):
        """
        生成每个特征层上的anchors
        :param per_level_anchors: 九组预设anchor中的三个
        :param feature_map_size: 输出的预测特征图[w, h]
        :param stride: [8, 16, 32]的其中一个
        :return:
        """
        # shifts_x shape:[w],shifts_x shape:[h]
        shift_x = [p for p in np.arange(0, feature_map_size.shape[0])]
        shift_y = [p for p in np.arange(0, feature_map_size.shape[1])]


if __name__ == "__main__":
    yolo_anchor = YoloV3Anchors()

