import os
import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms, batched_nms


class DetNMSMethod:
    def __init__(self, nms_type='python_nms', nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'Wrong nms type'

        self.nms_type = nms_type
        self.nms_threshold = nms_threshold

    def __call__(self, sorted_bboxes, sorted_scores, sorted_classes):
        """

        Args:
            sorted_bboxes: [anchor_num, 4], 4: x_min, y_min, x_max, y_max
            sorted_scores: [anchor_num], classification predict scores

        Returns:
        """
        if self.nms_type == 'torch_nms':
            sorted_bboxes = torch.tensor(sorted_bboxes, requires_grad=False)
            sorted_scores = torch.tensor(sorted_scores, requires_grad=False)
            sorted_classes = torch.tensor(sorted_classes, requires_grad=False)
            # keep = nms(sorted_bboxes, sorted_scores, self.nms_threshold)
            keep = batched_nms(sorted_bboxes, sorted_scores, sorted_classes, self.nms_threshold)
            keep = keep.cpu().detach().numpy()
        else:
            sorted_bboxes_wh = sorted_bboxes[:, 2:4] - sorted_bboxes[:, 0:2]  # [anchor_num, 2]
            sorted_bboxes_areas = sorted_bboxes_wh[:, 0] * sorted_bboxes_wh[:, 1]  # [anchor_num]

            # 保证sorted_bboxes_areas中的最小值是0
            sorted_bboxes_areas = np.maximum(sorted_bboxes_areas, 0)

            indexes = np.array([i for i in range(sorted_scores.shape[0])], dtype=np.int32)

            keep = []
            while indexes.shape[0] > 0:
                keep_idx = indexes[0]
                keep.append(keep_idx)
                indexes = indexes[1:]
                if len(indexes) == 0:
                    break

                keep_box_area = sorted_bboxes_areas[keep_idx]
                overlap_top_left = np.maximum(sorted_bboxes[keep_idx, 0:2], sorted_bboxes[indexes, 0:2])
                overlap_bot_right = np.minimum(sorted_bboxes[keep_idx, 2:4], sorted_bboxes[indexes, 2:4])
                overlap_area_sizes = np.maximum(overlap_bot_right - overlap_top_left, 0)
                overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

                # compute ious for top1 pred_bbox and the other pred_bboxes
                unions_area = sorted_bboxes_areas[indexes] + keep_box_area - overlap_area
                unions_area = np.maximum(unions_area, 1e-4)
                ious = overlap_area / unions_area

                if self.nms_type == 'diou_python_nms':
                    enclose_area_top_left = np.minimum(sorted_bboxes[indexes, 0:2], sorted_bboxes[keep_idx, 0:2])
                    enclose_area_bot_right = np.maximum(sorted_bboxes[indexes, 2:4], sorted_bboxes[keep_idx, 2:4])
                    # [N, 2]
                    enclose_area_sizes = np.maximum(enclose_area_bot_right - enclose_area_top_left, 0)
                    # [N]
                    c2 = (enclose_area_sizes ** 2).sum(axis=1)
                    c2 = np.maximum(c2, 1e-4)

                    keep_bbox_ctr = (sorted_bboxes[keep_idx, 0:2] + sorted_bboxes[keep_idx, 2:4]) / 2
                    other_bbox_ctr = (sorted_bboxes[indexes, 0:2] + sorted_bboxes[indexes, 2:4]) / 2
                    p2 = (other_bbox_ctr - keep_bbox_ctr) ** 2
                    p2 = np.sum(p2, axis=1)
                    ious = ious - p2 / c2
                # candidate_indexes = np.where(indexes, ious>self.nms_threshold)
                indexes = indexes[ious < self.nms_threshold]
            keep = np.array(keep)
        return keep


class DecodeMethod:
    def __init__(self,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        self.max_object_num = max_object_num
        self.min_score_threshold = min_score_threshold
        self.topn = topn
        self.nms_type = nms_type
        self.nms_threshold = nms_threshold
        self.nms_function = DetNMSMethod(nms_type=self.nms_type,
                                         nms_threshold=self.nms_threshold)

    def __call__(self, cls_scores, cls_classes, pred_bboxes):
        """

        Args:
            cls_scores: [B, N]
            cls_classes: [B, N]
            pred_bboxes: [B, N, 4]

        Returns:

        """
        batch_size = cls_scores.shape[0]
        batch_scores = np.ones((batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_classes = np.ones((batch_size, self.max_object_num), dtype=np.float32) * (-1)
        batch_bboxes = np.zeros((batch_size, self.max_object_num, 4), dtype=np.float32)

        for img_idx, (per_image_scores, per_image_score_classes, per_image_pred_bboxes) \
                in enumerate(zip(cls_scores, cls_classes, pred_bboxes)):
            mask = per_image_scores > self.min_score_threshold
            score_classes = per_image_score_classes[mask].astype(np.float32)
            bboxes = per_image_pred_bboxes[mask].astype(np.float32)
            scores = per_image_scores[mask].astype(np.float32)

            if scores.shape[0] > 0:
                # 从大到小排序
                sorted_indexes = np.argsort(-scores)
                sorted_scores = scores[sorted_indexes]
                sorted_score_classes = score_classes[sorted_indexes]
                sorted_bboxes = bboxes[sorted_indexes]

                if self.topn < sorted_scores.shape[0]:
                    sorted_scores = sorted_scores[0:self.topn]
                    sorted_score_classes = sorted_score_classes[0:self.topn]
                    sorted_bboxes = sorted_bboxes[0:self.topn]

                # nms
                keep = self.nms_function(sorted_bboxes, sorted_scores, sorted_score_classes)
                keep_scores = sorted_scores[keep]
                keep_classes = sorted_score_classes[keep]
                keep_bboxes = sorted_bboxes[keep]

                final_detection_num = min(self.max_object_num, keep_scores.shape[0])

                batch_scores[img_idx, :final_detection_num] = keep_scores[:final_detection_num]
                batch_classes[img_idx, :final_detection_num] = keep_classes[:final_detection_num]
                batch_bboxes[img_idx, :final_detection_num, :] = keep_bboxes[:final_detection_num, :]

        # batch_scores shape: [B, max_obj_num]
        # batch_scores shape: [B, max_obj_num]
        # batch_scores shape: [B, max_obj_num, 4]
        return [batch_scores, batch_classes, batch_bboxes]


if __name__ == "__main__":
    index = np.array([i + 5 for i in range(7)])
    print(index)
    ious = np.array([0.5849784, 0.66039437, 0., 0.57504936, 0.29496997, 0., 0.36191587])
    res = np.argsort(-ious)
    print(res)


