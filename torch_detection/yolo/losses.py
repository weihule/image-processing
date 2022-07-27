import os
import sys
import torch
import torch.nn as nn
from network_files.anchor import YoloV3Anchors

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from torch_detection.utils.iou_methos import IoUMethod


class YoloV5Loss(nn.Module):
    def __init__(self,
                 anchor_sizes=None,
                 strides=None,
                 per_level_num_anchors=3,
                 obj_layer_weight=None,
                 obj_loss_weight=1.,
                 box_loss_weight=0.05,
                 cls_loss_weight=0.5,
                 box_loss_iou_type='CIoU',
                 filter_anchor_threshold=4.):
        super(YoloV5Loss, self).__init__()
        assert box_loss_iou_type in ['IoU, DIoU, CIoU'], 'Wrong IoU type'
        if anchor_sizes is None:
            self.anchor_sizes = [[10, 13], [16, 30], [33, 23], [30, 61],
                            [62, 45], [59, 119], [116, 90], [156, 198],
                            [373, 326]]
        if strides is None:
            self.strides = [8, 16, 32]
        if obj_layer_weight is None:
            self.obj_layer_weight = [4.0, 1.0, 0.4]
        self.anchors = YoloV3Anchors(
            anchor_sizes=self.anchor_sizes,
            strides=self.strides,
            per_level_num_anchors=per_level_num_anchors
        )
        self.obj_loss_weight = obj_loss_weight
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_iou_type = box_loss_iou_type
        self.filter_anchor_threshold = filter_anchor_threshold
        self.iou_function = IoUMethod(iou_type=self.box_loss_iou_type)

    def forward(self, preds, annotations):
        """
        compute obj loss, reg loss and cls loss in one batch
        :param preds:
        :param annotations:
        :return:
        """
        device = annotations.device
        batch_size = annotations.shape[0]

        # if input size:[B,3,416,416]
        # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
        # obj_reg_cls_heads shape:[[B, 52, 52, 3, 85],[B, 26, 26, 3, 85],[B, 13, 13, 3, 85]]
        obj_reg_cls_preds = preds[0]

        # feature_size = [[h, w], ...]
        feature_size = [[per_level_cls_head[2], per_level_cls_head[1]] for per_level_cls_head in obj_reg_cls_preds]
        one_image_anchors = self.anchors()

