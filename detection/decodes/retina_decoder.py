import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode_method import DetNMSMethod, DecodeMethod

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_detection.models.anchors import RetinaAnchor

__all__ = [
    "RetinaDecoder"
]


class RetinaDecoder:
    def __init__(self,
                 areas=([32, 32], [64, 64], [128, 128], [256, 256], [512, 512]),
                 ratios=(0.5, 1, 2),
                 scales=(2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)),
                 strides=(8, 16, 32, 64, 128),
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5
                 ):
        assert nms_type in ['torch_nms',
                            'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        self.anchors = RetinaAnchor(areas=areas,
                                    ratios=ratios,
                                    scales=scales,
                                    strides=strides)
        self.decode_function = DecodeMethod(
            max_object_num=max_object_num,
            min_score_threshold=min_score_threshold,
            topn=topn,
            nms_type=nms_type,
            nms_threshold=nms_threshold)

    def __call__(self, preds):
        # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
        # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
        cls_preds, reg_preds = preds
        # [[w,h] ...]
        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in cls_preds]

        # [[80,80,9,4],[40,40,9,4],[20,20,9,4],[10,10,9,4],[5,5,9,4]]
        one_image_anchors = self.anchors(feature_size)

        # [N, anchor_num, num_classes]
        cls_preds = np.concatenate([
            per_cls_pred.cpu().detach().numpy().reshape(
                per_cls_pred.shape[0], -1, per_cls_pred.shape[-1]
            )
            for per_cls_pred in cls_preds
        ], axis=1)

        # [N, anchor_num, 4]
        reg_preds = np.concatenate([
            per_reg_pred.cpu().detach().numpy().reshape(
                per_reg_pred.shape[0], -1, per_reg_pred.shape[-1]
            )
            for per_reg_pred in reg_preds
        ], axis=1)

        # [anchor_num, 4]
        one_image_anchors = np.concatenate([
            per_level_anchor.reshape(-1, per_level_anchor.shape[-1])
            for per_level_anchor in one_image_anchors
        ], axis=0)

        # [N, anchor_num, 4]
        batch_anchors = np.repeat(np.expand_dims(one_image_anchors, axis=0),
                                  cls_preds.shape[0],
                                  axis=0)

        # [B, anchor_num]
        cls_classes = np.argmax(cls_preds, axis=2)

        # [B, anchor_num] 每个anchor一个类别置信度
        # 已经是之前每行的最大值，所以这里就是每个anchor代表的类别分数
        cls_scores = np.concatenate([
            np.expand_dims(per_image_preds[np.arange(per_image_preds.shape[0]),
                                           per_image_cls_classes],
                           axis=0)
            for per_image_preds, per_image_cls_classes in zip(
                cls_preds, cls_classes
            )
        ], axis=0)

        pred_bboxes = self.snap_txtytwth_to_x1y1x2y2(reg_preds, batch_anchors)
        [batch_scores, batch_classes, batch_bboxes] = \
            self.decode_function(cls_scores, cls_classes, pred_bboxes)

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]

    def snap_txtytwth_to_x1y1x2y2(self, reg_preds, anchors):
        """

        Args:
            reg_preds: [batch_size,anchor_nums,4] 4:[tx,ty,tw,th]
            anchors: [batch_size,anchor_nums,4] 4:[x_min,y_min,x_max,y_max]
        Returns:

        """
        anchors_wh = anchors[:, :, 2:4] - anchors[:, :, 0:2]
        anchors_ctr = anchors[:, :, 0:2] + 0.5 * anchors_wh

        pred_bboxes_wh = np.exp(reg_preds[:, :, 2:4]) * anchors_wh
        pred_bboxes_ctr = reg_preds[:, :, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_min_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_max_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = np.concatenate(
            [pred_bboxes_x_min_y_min, pred_bboxes_x_max_y_max], axis=-1)
        pred_bboxes = pred_bboxes.astype(np.int32)

        # pred bboxes shape:[batch,anchor_nums,4]
        return pred_bboxes


if __name__ == "__main__":
    device = torch.device("cuda")
    cls = [torch.randn(4, 80, 80, 9, 20).to(device),
           torch.randn(4, 40, 40, 9, 20).to(device),
           torch.randn(4, 20, 20, 9, 20).to(device),
           torch.randn(4, 10, 10, 9, 20).to(device),
           torch.randn(4, 5, 5, 9, 20).to(device)]
    reg = [torch.randn(4, 80, 80, 9, 4).to(device),
           torch.randn(4, 40, 40, 9, 4).to(device),
           torch.randn(4, 20, 20, 9, 4).to(device),
           torch.randn(4, 10, 10, 9, 4).to(device),
           torch.randn(4, 5, 5, 9, 4).to(device)]
    ret = RetinaDecoder()
    ret([cls, reg])
