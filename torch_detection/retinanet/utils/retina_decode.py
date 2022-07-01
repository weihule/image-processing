import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from torchvision.ops import batched_nms


class RetinaNetDecoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 min_score_threshold=0.05,
                 nms_threshold=0.5,
                 max_detection_num=100):
        super(RetinaNetDecoder, self).__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.min_score_threshold = min_score_threshold
        self.nms_threshold = nms_threshold
        self.max_detection_num = max_detection_num

    def snap_tx_ty_tw_th_to_x1_y1_x2_y2(self, reg_heads, anchors):
        """
        sanp reg heads to pred bboxes
        reg_heads: [anchor_num, 4]   tx,ty,tw,th
        anchors: [anchor_num, 4]     x_min,y_min,x_max,y_max
        """
        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

        factor = torch.tensor([0.1, 0.1, 0.2, 0.2])
        reg_heads = reg_heads * factor
        pred_bboxes_wh = torch.exp(reg_heads[:, 2:]) * anchors_wh
        pred_bboxes_ctr = reg_heads[:, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_y_min = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = torch.cat((pred_bboxes_x_y_min, pred_bboxes_x_y_min), dim=1)
        pred_bboxes = pred_bboxes.int()

        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2], max=self.image_w-1)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3], max=self.image_h-1)

        # pred_bboxes shape is [anchor_num, 4]
        return pred_bboxes

    def custom_nms(self, one_img_pred_bboxes, one_img_scores, one_img_classes):
        """
        :param one_img_pred_bboxes: [anchor_num, 4]  x_min,y_min,x_max,y_max
        :param one_img_scores: [anchor_num]    classification predict scores
        :param one_img_classes: [anchor_num]   class indices for predict scores
        :return:
        """
