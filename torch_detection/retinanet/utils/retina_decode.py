import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
from torchvision.ops import batched_nms

from .losses import compute_ious_for_one_image


# sys.path.append(os.path.)


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

    # 这里输入的是5个feature map的结果
    # 传入的是list形式，里面有5个元素, 每个元素一个三维向量
    # 以cls_heads为例: [[B, f1_anchor_num, 80], [B, f2_anchor_num, 80], ...]
    def forward(self, cls_heads, reg_heads, batch_anchors):
        with torch.no_grad():
            devices = cls_heads[0].device
            batch_scores = list()
            batch_classes = list()
            batch_pred_bboxes = list()

            # 把该batch中的所有样本(anchor)全部合并,所以是沿着dim=1维拼接的
            cls_heads = torch.cat(cls_heads, dim=1)     # [B, f1+...+f5_anchor_num, num_classes]
            reg_heads = torch.cat(reg_heads, dim=1)
            batch_anchors = torch.cat(batch_anchors, dim=1)

            for per_img_cls_heads, per_img_reg_heads, per_img_anchors in zip(
                    cls_heads, reg_heads, batch_anchors):

                pred_bboxes = self.snap_tx_ty_tw_th_to_x1_y1_x2_y2(per_img_reg_heads, per_img_anchors)
                scores, scores_classes = torch.max(per_img_cls_heads, dim=1)

                mask = scores > self.min_score_threshold
                pred_bboxes = pred_bboxes[mask].float()
                scores_classes = scores_classes[mask].float()
                scores = scores[mask].float()

                single_img_scores = torch.ones((self.max_detection_num,), device=devices) * (-1)
                single_img_classes = torch.ones((self.max_detection_num,), device=devices) * (-1)
                single_img_pred_bboxes = torch.ones((self.max_detection_num, 4), device=devices) * (-1)

                if scores.shape[0] != 0:
                    scores, scores_classes, pred_bboxes = self.custom_batched_nms(pred_bboxes, scores, scores_classes)

                    sorted_keep_scores, sorted_keep_scores_indices = torch.sort(scores, descending=True)
                    sorted_keep_classes = scores_classes[sorted_keep_scores_indices]
                    sorted_keep_pred_bboxes = pred_bboxes[sorted_keep_scores_indices]

                    final_detection_num = min(self.max_detection_num, sorted_keep_scores.shape[0])

                    single_img_scores[0: final_detection_num] = sorted_keep_scores[0: final_detection_num]
                    single_img_classes[0: final_detection_num] = sorted_keep_classes[0: final_detection_num]
                    single_img_pred_bboxes[0: final_detection_num, :] = sorted_keep_pred_bboxes[0: final_detection_num, :]
                single_img_scores = torch.unsqueeze(single_img_scores, dim=0)  # [1, 100]
                single_img_classes = torch.unsqueeze(single_img_classes, dim=0)  # [1, 100]
                single_img_pred_bboxes = torch.unsqueeze(single_img_pred_bboxes, dim=0)  # [1, 100, 4]

                batch_scores.append(single_img_scores)
                batch_classes.append(single_img_classes)
                batch_pred_bboxes.append(single_img_pred_bboxes)
            batch_scores = torch.cat(batch_scores, dim=0)
            batch_classes = torch.cat(batch_classes, dim=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, dim=0)

            # batch_scores : [B, 100]
            # batch_classes : [B, 100]
            # batch_pred_bboxes : [B, 100, 4]

            return batch_scores, batch_classes, batch_pred_bboxes

    def snap_tx_ty_tw_th_to_x1_y1_x2_y2(self, reg_heads, anchors):
        """
        sanp reg heads to pred bboxes
        reg_heads: [anchor_num, 4]   tx,ty,tw,th
        anchors: [anchor_num, 4]     x_min,y_min,x_max,y_max
        """
        if reg_heads.shape[1] != 4 and anchors.shape[1] != 4:
            raise ValueError('shape expected anchor_num,4, but got {}'.format(reg_heads.shape))

        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

        devices = reg_heads.device

        factor = torch.tensor([0.1, 0.1, 0.2, 0.2], device=devices)
        reg_heads = reg_heads * factor
        pred_bboxes_wh = torch.exp(reg_heads[:, 2:]).to(devices) * anchors_wh.to(devices)
        pred_bboxes_ctr = reg_heads[:, :2] * anchors_wh + anchors_ctr

        pred_bboxes_x_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh

        pred_bboxes = torch.cat((pred_bboxes_x_y_min, pred_bboxes_x_y_max), dim=1)
        pred_bboxes = pred_bboxes.int()

        pred_bboxes[:, 0] = torch.clamp(pred_bboxes[:, 0], min=0)
        pred_bboxes[:, 1] = torch.clamp(pred_bboxes[:, 1], min=0)
        pred_bboxes[:, 2] = torch.clamp(pred_bboxes[:, 2], max=self.image_w - 1)
        pred_bboxes[:, 3] = torch.clamp(pred_bboxes[:, 3], max=self.image_h - 1)

        # pred_bboxes shape is [anchor_num, 4]
        return pred_bboxes

    def custom_batched_nms(self, one_img_pred_bboxes, one_img_scores, one_img_classes):
        """
        one_img_pred_bboxes: [anchor_num, 4]  x_min,y_min,x_max,y_max
        one_img_scores: [anchor_num]    classification predict scores
        one_img_classes: [anchor_num]   class indices for predict scores
        """
        # sort boxes
        sorted_scores, sorted_indices = torch.sort(one_img_scores, descending=True)
        sorted_classes = one_img_classes[sorted_indices]
        sorted_pred_bboxes = one_img_pred_bboxes[sorted_indices]  # [anchor_num, 4]
        sorted_pred_bboxes_wh = sorted_pred_bboxes[:, 2:] - sorted_pred_bboxes[:, :2]  # [anchor_num, 2]
        sorted_pred_bboxes_areas = sorted_pred_bboxes_wh[:, 0] * sorted_pred_bboxes_wh[:, 1]
        detected_classes = torch.unique(sorted_classes, sorted=True)

        keep_scores, keep_classes, keep_pred_bboxes = list(), list(), list()
        for detected_class in detected_classes:
            single_mask = torch.eq(sorted_classes, detected_class)
            single_class_scores = sorted_scores[single_mask]
            single_classes = sorted_classes[single_mask]
            single_pred_bboxes = sorted_pred_bboxes[single_mask]
            single_pred_bboxes_areas = sorted_pred_bboxes_areas[single_mask]

            single_keep_scores, single_keep_classes, single_keep_pred_bboxes = list(), list(), list()
            while single_class_scores.numel() > 0:
                top1_score = single_class_scores[0]
                # top1_class = single_classes[0]
                top1_pred_box = single_pred_bboxes[0]

                # single_keep_scores.append(top1_score)
                # single_keep_classes.append(detected_class)
                # single_keep_pred_bboxes.append(top1_pred_box)
                keep_scores.append(top1_score)
                keep_classes.append(detected_class)
                keep_pred_bboxes.append(top1_pred_box)

                top1_areas = single_pred_bboxes_areas[0]

                if single_class_scores.numel() == 1:
                    break
                single_class_scores = single_class_scores[1:]
                single_pred_bboxes = single_pred_bboxes[1:]

                ious = compute_ious_for_one_image(single_pred_bboxes.reshape(-1, 4), top1_pred_box.reshape(-1, 4))
                ious = ious.flatten()

                hidden_mask = ious < self.nms_threshold
                single_class_scores = single_class_scores[hidden_mask]
                single_pred_bboxes = single_pred_bboxes[hidden_mask]
            # keep_scores.append(single_keep_scores)
            # keep_classes.append(single_keep_classes)
            # keep_pred_bboxes.append(single_keep_pred_bboxes)
        # keep_scores = torch.cat()

        keep_scores = torch.tensor(keep_scores)
        keep_classes = torch.tensor(keep_classes)
        keep_pred_bboxes = torch.cat(keep_pred_bboxes).reshape((-1, 4))

        return keep_scores, keep_classes, keep_pred_bboxes


if __name__ == "__main__":
    # arr = torch.tensor([1, 9, 0, 2, 1, 5, 8, 3, 8, 6])
    # res = torch.unique(arr, sorted=True)
    # print(res)

    # classes = torch.randint(low=1, high=20, size=(23, ))
    # in_use_indices = res = torch.unique(classes, sorted=True)
    # print(classes, classes.shape)
    #
    #
    # for index in in_use_indices:
    #     print(index)
    #     mask = torch.eq(classes, index)
    #     print(mask)
    #     break

    # all_list = list()
    # all_list.append(torch.tensor(0))
    # all_list.append(torch.tensor(1))
    # all_list.append(torch.tensor(2))
    # all_list.append(torch.tensor(3))
    # all_list.append(torch.tensor(4))
    #
    # res = torch.tensor(all_list)
    # print(res, res.shape)

    arr1 = torch.rand(1, 4)
    arr2 = torch.rand(1, 4)
    arr_list = [arr1, arr2]
    arr = torch.cat(arr_list, dim=0)
    print(arr, arr.shape)
