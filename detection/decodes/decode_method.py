import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import nms

__all__ = [
    "DecodeMethod",
    "DetNMSMethod"
]


class DetNMSMethod:
    def __init__(self, nms_type='python_nms', nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        self.nms_type = nms_type
        self.nms_threshold = nms_threshold

    def __call__(self, sorted_bboxes, sorted_scores):
        """
        sorted_bboxes:[anchor_nums,4], 4:x_min, y_min, x_max, y_max
        sorted_scores:[anchor_nums], classification predict scores
        """
        if self.nms_type == "python_nms":
            keep = self.my_nms(sorted_bboxes, sorted_scores, self.nms_threshold)
        else:
            keep = self.torch_nms(sorted_bboxes, sorted_scores, self.nms_threshold)

        return keep

    @staticmethod
    def torch_nms(sorted_bboxes, sorted_scores, nms_threshold):
        sorted_bboxes, sorted_scores = torch.tensor(sorted_bboxes).cpu(
        ).detach(), torch.tensor(sorted_scores).cpu().detach()
        keep = nms(sorted_bboxes, sorted_scores, nms_threshold)
        keep = keep.cpu().detach().numpy()

        return keep

    @staticmethod
    def my_nms(sorted_bboxes, sorted_scores, nms_threshold):
        """
        不分类别, 对所有的框进行nms
        """
        # [anchor_num, 2]
        sorted_bboxes_wh = sorted_bboxes[:, 2:4] - sorted_bboxes[:, 0:2]
        # [anchor_num, ]
        sorted_bboxes_areas = sorted_bboxes_wh[:, 0] * sorted_bboxes_wh[:, 1]
        # [anchor_num, ]
        sorted_bboxes_areas = np.maximum(sorted_bboxes_areas, 0)

        indexes = np.array([i for i in range(sorted_scores.shape[0])],
                           dtype=np.int32)
        keep = []
        while indexes.shape[0] > 0:
            keep_idx = indexes[0]
            keep.append(keep_idx)
            indexes = indexes[1:]
            if len(indexes) == 0:
                break
            keep_box_area = sorted_bboxes_areas[keep_idx]

            # 计算重叠部分的四点
            overlap_area_top_left = np.maximum(
                sorted_bboxes[keep_idx, 0:2], sorted_bboxes[indexes, 0:2])
            overlap_area_bot_right = np.minimum(
                sorted_bboxes[keep_idx, 2:4], sorted_bboxes[indexes, 2:4])
            overlap_area_sizes = np.maximum(
                overlap_area_bot_right - overlap_area_top_left, 0)
            overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

            # 计算top1和其他框之间的iou
            union_area = keep_box_area + sorted_bboxes_areas[indexes] - overlap_area
            union_area = np.maximum(union_area, 1e-4)
            ious = overlap_area / union_area

            # np.where(condition)
            # 这种用法返回满足条件的元素的索引。
            # 索引以元组的形式返回，其中包含满足条件的元素的行索引和列索引（对于多维数组）
            # 所以这里取[0], 直接取行索引
            candidate_indexes = np.where(ious < nms_threshold)[0]
            indexes = indexes[candidate_indexes]
        keep = np.array(keep)

        return keep


class DecodeMethod:
    def __init__(self,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type="python_nms",
                 nms_threshold=0.5):
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_function = DetNMSMethod(nms_type=nms_type,
                                         nms_threshold=nms_threshold)

    def __call__(self, cls_scores, cls_classes, pred_bboxes):
        """

        Args:
            cls_scores: [B, anchor_num]   每个anchor表示的类别置信度
            cls_classes: [B, anchor_num]  每个anchor表示的类别
            pred_bboxes: [B, anchor_num, 4]

        Returns:

        """
        batch_size = cls_scores.shape[0]
        batch_scores = np.ones((batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_classes = np.ones((batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_bboxes = np.zeros((batch_size, self.max_object_num, 4), dtype=np.float32)
        for i, (per_image_scores,
                per_image_score_classes,
                per_image_pred_bboxes) in enumerate(zip(cls_scores, cls_classes, pred_bboxes)):
            score_classes = per_image_score_classes[per_image_scores > self.min_score_threshold].astype(np.float32)
            bboxes = per_image_pred_bboxes[per_image_scores > self.min_score_threshold].astype(np.float32)
            scores = per_image_scores[per_image_scores > self.min_score_threshold].astype(np.float32)
            # score_classes: [anchor_num, num_classes]
            # bboxes: [anchor_num, 4]
            # scores: [anchor_num]
            # 注意这里的anchor_num指的是通过阈值过滤后的数量

            if scores.shape[0] != 0:
                # descending sort
                sorted_indexes = np.argsort(-scores)
                sorted_scores = scores[sorted_indexes]
                sorted_score_classes = score_classes[sorted_indexes]
                sorted_bboxes = bboxes[sorted_indexes]

                if sorted_scores.shape[0] > self.topn:
                    sorted_scores = sorted_scores[0:self.topn]
                    sorted_score_classes = sorted_score_classes[0:self.topn]
                    sorted_bboxes = sorted_bboxes[0:self.topn]

                # NMS
                keep = self.nms_function(sorted_bboxes, sorted_scores)
                keep_scores = sorted_scores[keep]
                keep_classes = sorted_score_classes[keep]
                keep_bboxes = sorted_bboxes[keep]

                final_detection_num = min(self.max_object_num, keep_scores.shape[0])
                batch_scores[i, 0:final_detection_num] = keep_scores[0:final_detection_num]
                batch_classes[i, 0:final_detection_num] = keep_classes[0:final_detection_num]
                batch_bboxes[i, 0:final_detection_num, :] = keep_bboxes[0:final_detection_num, :]

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        return [batch_scores, batch_classes, batch_bboxes]


