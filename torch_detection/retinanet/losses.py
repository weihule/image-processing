import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_detection.utils.iou_method import IoUMethodMultiple
from torch_detection.retinanet.network_files.anchors import RetinaAnchors


class RetinaLoss(nn.Module):
    def __init__(self,
                 areas=None,
                 rations=None,
                 scales=None,
                 strides=None,
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 focal_eiou_gamma=0.5,
                 cls_loss_weight=1.,
                 box_loss_weight=1.,
                 box_loss_type='CIoU'):
        super(RetinaLoss, self).__init__()
        if areas is None:
            self.areas = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]],
        else:
            self.areas = areas

        if rations is None:
            self.ratios = [0.5, 1, 2]
        else:
            self.ratios = rations

        if scales is None:
            self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        else:
            self.scales = scales

        if strides is None:
            self.strides = [8, 16, 32, 64, 128]
        else:
            self.strides = strides
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.focal_eiou_gamma = focal_eiou_gamma
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_weight = box_loss_weight
        self.box_loss_type = box_loss_type
        if self.box_loss_type not in ['SmoothL1', 'IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU', 'Focal_EIoU']:
            raise ValueError('Wrong IoU type')
        self.anchors = RetinaAnchors()
        self.iou_function = IoUMethodMultiple()

    def forward(self, preds, annotations):
        """
        compute cls loss and reg loss in one batch
        :param preds: [cls_heads, reg_heads]
                cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80], ...]
                reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4], ...]
        :param annotations: [B, num, 5]
        :return:
        """
        device = annotations.device
        batch_size = annotations.shape[0]
        cls_preds, reg_preds = preds

        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]] for per_level_cls_pred in cls_preds]

        # one_image_anchors shape [h, w, 9, 4]  [[80, 80, 9, 4], [40, 40, 9, 4], ...]
        one_image_anchors = self.anchors(feature_size)

        # one_image_anchors shape [h1*w1*9+h2*w2*9+..., 4]
        one_image_anchors = torch.cat([
            torch.tensor(p).view(-1, p.shape[-1]) for p in one_image_anchors], dim=0)

        # batch_anchors shape [B, h1*w1*9+h2*w2*9+..., 4]
        batch_anchors = one_image_anchors.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        batch_anchors_annotations = self.get_batch_anchors_annotations(
            batch_anchors, annotations
        )

        cls_preds = [per_cls_pred.view(per_cls_pred.shape[0], -1, per_cls_pred.shape[-1])
                     for per_cls_pred in cls_preds]
        reg_preds = [per_reg_pred.view(per_reg_pred.shape[0], -1, per_reg_pred.shape[-1])
                     for per_reg_pred in reg_preds]
        cls_preds = torch.cat(cls_preds, dim=1)  # [B, h1*w1*9+..., 80]
        reg_preds = torch.cat(reg_preds, dim=1)  # [B, h1*w1*9+..., 4]

        cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)

        cls_preds = cls_preds.view(-1, cls_preds.shape[-1])  # [B*h1*w1*9+..., 80]
        reg_preds = reg_preds.view(-1, reg_preds.shape[-1])  # [B*h1*w1*9+..., 4]
        batch_anchors = batch_anchors.view(-1, batch_anchors.shape[-1])  # [B*h1*w1*9+..., 4]

        batch_anchors_annotations = batch_anchors_annotations.view(-1, batch_anchors_annotations.shape[-1])

        cls_loss = self.compute_batch_focal_loss(cls_preds,
                                                 batch_anchors_annotations)
        reg_loss = self.compute_batch_box_loss(reg_preds,
                                               batch_anchors_annotations,
                                               batch_anchors)

        cls_loss = self.cls_loss_weight * cls_loss
        reg_loss = self.box_loss_weight * reg_loss

        loss_dict = {
            'cls_loss': cls_loss,
            'reg_loss': reg_loss,
        }

        return loss_dict

    def compute_batch_focal_loss(self, cls_preds, batch_anchors_annotations):
        """
        compute batch focal loss(cls loss)
        cls_preds: [batch_size*anchor_num,num_classes]
        batch_anchors_annotations: [batch_size*anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        device = cls_preds.device
        mask = batch_anchors_annotations[:, 4] >= 0
        cls_preds = cls_preds[mask]
        batch_anchors_annotations = batch_anchors_annotations[mask]
        positive_anchors_num = batch_anchors_annotations[batch_anchors_annotations[:, 4] > 0].shape[0]

        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device)

        num_classes = cls_preds.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(batch_anchors_annotations[:, 4].long(),
                                      num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth[:, 1:]
        loss_ground_truth = loss_ground_truth.float()

        alpha_factor = torch.ones_like(cls_preds) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.),
                                   alpha_factor, 1. - alpha_factor)
        pt = torch.where(torch.eq(loss_ground_truth, 1.), cls_preds,
                         1. - cls_preds)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        batch_bce_loss = -(
                loss_ground_truth * torch.log(cls_preds) +
                (1. - loss_ground_truth) * torch.log(1. - cls_preds))

        batch_focal_loss = focal_weight * batch_bce_loss
        batch_focal_loss = batch_focal_loss.sum()
        # according to the original paper,We divide the focal loss by the number of positive sample anchors
        batch_focal_loss = batch_focal_loss / positive_anchors_num

        return batch_focal_loss

    def compute_batch_box_loss(self, reg_preds, batch_anchors_annotations,
                               batch_anchors):
        """
        compute batch smoothl1 loss(reg loss)
        reg_preds:[batch_size*anchor_num,4]
        batch_anchors_annotations:[batch_size*anchor_num,5]
        batch_anchors:[batch_size*anchor_num,4]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate smoothl1 loss
        device = reg_preds.device
        reg_preds = reg_preds[batch_anchors_annotations[:, 4] > 0]
        batch_anchors = batch_anchors[batch_anchors_annotations[:, 4] > 0]
        batch_anchors_annotations = batch_anchors_annotations[batch_anchors_annotations[:, 4] > 0]
        positive_anchor_num = batch_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        if self.box_loss_type == 'SmoothL1':
            box_loss = self.compute_batch_smoothl1_loss(
                reg_preds, batch_anchors_annotations)
        else:
            box_loss_type = 'EIoU' if self.box_loss_type == 'Focal_EIoU' else self.box_loss_type
            pred_boxes = self.snap_txtytwth_to_xyxy(reg_preds, batch_anchors)   # [batch_size*anchor_num, 4]
            # TODO: 这块相当于是每个pred_bbox和对应的gt做iou，这两部分的框的数量都是batch_size*anchor_num
            ious = self.iou_function(pred_boxes,
                                     batch_anchors_annotations[:, 0:4],
                                     iou_type=box_loss_type,
                                     box_type='xyxy')
            box_loss = 1 - ious

            if self.box_loss_type == 'Focal_EIoU':
                gamma_ious = self.iou_function(pred_boxes,
                                               batch_anchors_annotations[:, 0:4],
                                               iou_type='IoU',
                                               box_type='xyxy')
                gamma_ious = torch.pow(gamma_ious, self.focal_eiou_gamma)
                box_loss = gamma_ious * box_loss

            box_loss = box_loss.sum() / positive_anchor_num

        return box_loss

    def compute_batch_smoothl1_loss(self, reg_preds,
                                    batch_anchors_annotations):
        """
        compute batch smoothl1 loss(reg loss)
        reg_preds:[batch_size*anchor_num,4]
        anchors_annotations:[batch_size*anchor_num,5]
        """
        device = reg_preds.device
        positive_anchor_num = batch_anchors_annotations.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        # compute smoothl1 loss
        loss_ground_truth = batch_anchors_annotations[:, 0:4]

        x = torch.abs(reg_preds - loss_ground_truth)
        batch_smoothl1_loss = torch.where(torch.ge(x, self.beta),
                                          x - 0.5 * self.beta,
                                          0.5 * (x ** 2) / self.beta)

        batch_smoothl1_loss = batch_smoothl1_loss.sum() / positive_anchor_num

        return batch_smoothl1_loss

    def get_batch_anchors_annotations(self, batch_anchors, annotations):
        """
        :param batch_anchors: [B, h1*w1*9+h2*w2*9+..., 4]
        :param annotations: [B, gt_num, 5]
        Assign a ground truth box target and a ground truth class target for each anchor
        if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
        if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
        if anchor gt_class index > 0,this anchor is an object class anchor and used in
        calculate cls loss and reg loss
        return:
            [B, h1*w1*9+..., 5]
            相当于是将原来的 gt_num 个gt bbox变成和每一个anchor匹配的gt bbox
            并且打标签，背景的gt的标签为0, 前景的gt标签为原来的gt_label+1, 其余的gt标签为-1
        """
        if batch_anchors.shape[0] != annotations.shape[0]:
            raise ValueError('batch_size is not equal')
        device = annotations.device
        one_image_anchor_nums = batch_anchors.shape[1]

        batch_anchors_annotations = []

        # per_img_anchors shape is [h1*w1*9+..., 4]
        # per_img_annotations shape is [gt_num, 4]
        for per_img_anchors, per_img_annotations in zip(batch_anchors, annotations):
            # drop all index = -1 class annots
            per_img_annotations = per_img_annotations[per_img_annotations[:, 4] >= 0]
            if per_img_annotations.shape[0] == 0:

                # per_image_anchor_annotations shape is [h1*w1*9+h2*w2*9+..., 5]
                per_image_anchor_annotations = torch.ones((one_image_anchor_nums, 5),
                                                          dtype=torch.float32,
                                                          device=device) * (-1)
            else:
                per_img_gt_bboxes = per_img_annotations[:, :4]  # [per_img_gt_num, 4]
                per_img_gt_class = per_img_annotations[:, 4]  # [per_img_gt_num, ]

                # one_image_ious shape [h1*w1*9+h2*w2*9+..., per_img_gt_num]
                one_image_ious = self.iou_function(
                    per_img_anchors.unsqueeze(1),
                    per_img_gt_bboxes.unsqueeze(0),
                    iou_type='IoU',
                    box_type='xyxy')

                # snap per gt bboxes to the best iou anchor
                overlap, indices = one_image_ious.max(dim=1)  # [h1*w1*9+h2*w2*9+..., ]
                per_image_anchor_gt_class = torch.ones(overlap.shape[0],
                                                       dtype=torch.float32,
                                                       device=device)*(-1)

                # if iou < 0.4, assign anchors gt class as 0:background
                per_image_anchor_gt_class[overlap < 0.4] = 0

                # if iou >= 0.5
                # assign anchors gt class as same as the max iou annotation class:80 classes index from 1 to 80(20)
                per_image_anchor_gt_class[overlap >= 0.5] = per_img_gt_class[indices[overlap >= 0.5]] + 1

                per_image_anchor_gt_class = per_image_anchor_gt_class.unsqueeze(-1)  # [h1*w1*9+h2*w2*9+..., 1]

                # assign each anchor gt bboxes for max iou annotation
                per_image_anchor_gt_bboxes = per_img_gt_bboxes[indices]

                # [h1*w1*9+h2*w2*9+..., 5]
                # 其中包含背景(class=0), 无效anchor(class=-1)
                per_image_anchor_annotations = torch.cat((
                    per_image_anchor_gt_bboxes, per_image_anchor_gt_class), dim=1)
            # [1, h1*w1*9+h2*w2*9+..., 5]
            per_image_anchor_annotations = per_image_anchor_annotations.unsqueeze(0)
            batch_anchors_annotations.append(per_image_anchor_annotations)

        # [B, h1*w1*9+h2*w2*9+..., 5]
        batch_anchors_annotations = torch.cat(batch_anchors_annotations, dim=0)

        return batch_anchors_annotations

    @staticmethod
    def snap_annotations_to_txtytwth(anchors_gt_bboxes, anchors):
        """
        snap each anchor ground truth bbox form format:[x_min,y_min,x_max,y_max] to format:[tx,ty,tw,th]
        """
        anchors_w_h = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_w_h

        anchors_gt_bboxes_w_h = anchors_gt_bboxes[:, 2:] - anchors_gt_bboxes[:, :2]
        anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1e-4)
        anchors_gt_bboxes_ctr = anchors_gt_bboxes[:, :2] + 0.5 * anchors_gt_bboxes_w_h

        snaped_annotations_for_anchors = torch.cat(
            [(anchors_gt_bboxes_ctr - anchors_ctr) / anchors_w_h,
             torch.log(anchors_gt_bboxes_w_h / anchors_w_h)],
            dim=1)

        # snaped_annotations_for_anchors shape:[anchor_nums, 4]
        return snaped_annotations_for_anchors

    @staticmethod
    def snap_txtytwth_to_xyxy(snap_boxes, anchors):
        """
        snap reg heads to pred bboxes
        snap_boxes:[batch_size*anchor_nums,4],4:[tx,ty,tw,th], 这里就是 reg_preds
        anchors:[batch_size*anchor_nums,4],4:[x_min,y_min,x_max,y_max]
        """
        anchors_wh = anchors[:, 2:4] - anchors[:, 0:2]
        anchors_ctr = anchors[:, 0:2] + 0.5 * anchors_wh

        boxes_wh = torch.exp(snap_boxes[:, 2:4]) * anchors_wh
        boxes_ctr = snap_boxes[:, :2] * anchors_wh + anchors_ctr

        boxes_x_min_y_min = boxes_ctr - 0.5 * boxes_wh
        boxes_x_max_y_max = boxes_ctr + 0.5 * boxes_wh

        boxes = torch.cat([boxes_x_min_y_min, boxes_x_max_y_max], dim=1)

        # boxes shape:[anchor_nums,4]
        return boxes


if __name__ == "__main__":
    pred_bbs = torch.tensor([[100, 140, 120, 234], [5, 2, 10, 9], [7, 4, 12, 12], [17, 14, 20, 18],
                             [6, 14, 12, 18], [8, 9, 14, 15], [2, 20, 5, 25], [11, 7, 15, 16],
                             [8, 6, 13, 14], [10, 10, 14, 16], [10, 8, 14, 16], [18, 20, 40, 38]], dtype=torch.float32)
    gt_bbs = torch.tensor([[9, 8, 14, 15], [6, 5, 13, 11]], dtype=torch.float32)
    rl = RetinaLoss(32, 32)
    # res = snap_annotations_as_tx_ty_tw_th(gt_bbs, pred_bbs)
