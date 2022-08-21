import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import batched_nms

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from network_files.anchors import RetinaAnchors
from torch_detection.utils.iou_method import IoUMethod, IoUMethodNumpy, IoUMethodMultiple
from torch_detection.utils.decode_util import DecodeMethod


class RetinaDecoder:

    def __init__(self,
                 areas=None,
                 ratios=None,
                 scales=None,
                 strides=None,
                 max_object_num=100,
                 min_score_threshold=0.05,
                 topn=1000,
                 nms_type='python_nms',
                 nms_threshold=0.5):
        assert nms_type in ['torch_nms', 'python_nms',
                            'diou_python_nms'], 'wrong nms type!'
        if areas is None:
            self.areas = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]
        else:
            self.areas = areas

        if ratios is None:
            self.ratios = [0.5, 1, 2]
        else:
            self.ratios = ratios

        if scales is None:
            self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        else:
            self.scales = scales

        if strides is None:
            self.strides = [8, 16, 32, 64, 128]
        else:
            self.strides = strides
        self.anchors = RetinaAnchors(areas=self.areas,
                                     ratios=self.ratios,
                                     scales=self.scales,
                                     strides=self.strides)
        self.decode_func = DecodeMethod(max_object_num=max_object_num,
                                        min_score_threshold=min_score_threshold,
                                        topn=topn,
                                        nms_type=nms_type,
                                        nms_threshold=nms_threshold)

    def __call__(self, preds):
        # if input size:[B,3,640,640]
        # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
        # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
        cls_preds, reg_preds = preds

        # [[w, h], ...]
        feature_size = [[p.shape[2], p.shape[1]] for p in cls_preds]

        # if input size: [640, 640]
        # [h, w, 9, 4]
        # one_image_anchors shape: [[80, 80, 9, 4], [40, 40, 9, 4], ...]
        one_image_anchors = self.anchors(feature_size)

        # [B, h1*w1*9+..., 80]
        cls_preds = np.concatenate([
            p.cpu().detach().numpy().reshape(p.shape[0], -1, p.shape[-1])
            for p in cls_preds], axis=1)

        # [B, h1*w1*9+..., 4]
        reg_preds = np.concatenate([
            p.cpu().detach().numpy().reshape(p.shape[0], -1, p.shape[-1])
            for p in reg_preds], axis=1)

        # [h1*w1*9+..., 4]
        one_image_anchors = np.concatenate([
            p.reshape(-1, p.shape[-1]) for p in one_image_anchors])

        # [B, h1*w1*9+..., 4]
        batch_anchors = np.tile(one_image_anchors, (cls_preds.shape[0], 1, 1))

        # [B, h1*w1*9+...]
        cls_classes = np.argmax(cls_preds, axis=2)

        # TODO 这里是找出每一行最大值的索引，并将该位置的元素取出来, 在torch中直接 torch.max() 即可
        # per_image_preds [h1*w1*9+..., 80]
        # per_image_cls_classes中的数值范围是 [0, 80), 共有 h1*w1*9+... 个
        # cls_socres [B, h1*w1*9+...]
        cls_socres = np.concatenate([
            np.expand_dims(per_image_preds[np.arange(per_image_preds.shape[0]), per_image_cls_classes], axis=0)
            for per_image_preds, per_image_cls_classes in zip(cls_preds, cls_classes)
        ])

        # pred_bboxes [B, h1*w1*9+..., 4]
        pred_bboxes = self.sanp_txtytwth_to_x1y1x2y2(reg_preds, batch_anchors)
        [batch_scores, batch_classes, batch_bboxes] = self.decode_func(cls_scores=cls_socres,
                                                                       cls_classes=cls_classes,
                                                                       pred_bboxes=pred_bboxes)
        return [batch_scores, batch_classes, batch_bboxes]

    def sanp_txtytwth_to_x1y1x2y2(self, reg_preds, batch_anchors):
        """
        snap reg heads to pred heads
        Args:
            reg_preds: [B, anchor_num, 4]  4: [tx, ty, tw, th]
            batch_anchors: [B, anchor_num, 4]  4: [x1, y1, x2, y2]
        Returns:
        """
        anchors_wh = batch_anchors[..., 2:] - batch_anchors[..., :2]
        anchors_ctr = batch_anchors[..., :2] + anchors_wh * 0.5

        pred_bboxes_wh = np.exp(reg_preds[..., 2:4]) * anchors_wh
        pred_bboxes_ctr = reg_preds[..., 0:2] * anchors_ctr + anchors_ctr
        pred_bboxes_x_min_y_min = pred_bboxes_ctr - 0.5 * pred_bboxes_wh
        pred_bboxes_x_max_y_max = pred_bboxes_ctr + 0.5 * pred_bboxes_wh
        pred_bboxes = np.concatenate([
            pred_bboxes_x_min_y_min, pred_bboxes_x_max_y_max], axis=-1)
        pred_bboxes = pred_bboxes.astype(np.int32)

        return pred_bboxes


class RetinaNetDecoder(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 min_score_threshold=0.1,
                 nms_threshold=0.5,
                 max_detection_num=50):
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            devices = cls_heads[0].device
            batch_scores = list()
            batch_classes = list()
            batch_pred_bboxes = list()

            # 把该batch中每张图片的所有样本(anchor)全部合并,所以是沿着dim=1维拼接的
            cls_heads = torch.cat(cls_heads, dim=1)  # [B, f1+...+f5_anchor_num, num_classes]
            reg_heads = torch.cat(reg_heads, dim=1)
            batch_anchors = torch.cat(batch_anchors, dim=1).to(device)

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
                    indices = batched_nms(pred_bboxes, scores, scores_classes, 0.5)
                    scores, scores_classes, pred_bboxes = scores[indices], scores_classes[indices], pred_bboxes[indices]
                    # scores, scores_classes, pred_bboxes = self.custom_batched_nms(pred_bboxes, scores, scores_classes)

                    # sorted_keep_scores, sorted_keep_scores_indices = torch.sort(scores, descending=True)
                    # sorted_keep_classes = scores_classes[sorted_keep_scores_indices]
                    # sorted_keep_pred_bboxes = pred_bboxes[sorted_keep_scores_indices]

                    final_detection_num = min(self.max_detection_num, scores.shape[0])

                    single_img_scores[0: final_detection_num] = scores[0: final_detection_num]
                    single_img_classes[0: final_detection_num] = scores_classes[0: final_detection_num]
                    single_img_pred_bboxes[0: final_detection_num, :] = pred_bboxes[0: final_detection_num, :]
                single_img_scores = torch.unsqueeze(single_img_scores, dim=0)  # [1, max_detection_num]
                single_img_classes = torch.unsqueeze(single_img_classes, dim=0)  # [1, max_detection_num]
                single_img_pred_bboxes = torch.unsqueeze(single_img_pred_bboxes, dim=0)  # [1, max_detection_num, 4]

                batch_scores.append(single_img_scores)
                batch_classes.append(single_img_classes)
                batch_pred_bboxes.append(single_img_pred_bboxes)
            batch_scores = torch.cat(batch_scores, dim=0)
            batch_classes = torch.cat(batch_classes, dim=0)
            batch_pred_bboxes = torch.cat(batch_pred_bboxes, dim=0)

            # batch_scores : [B, final_detection_num]
            # batch_classes : [B, final_detection_num]
            # batch_pred_bboxes : [B, final_detection_num, 4]

            return batch_scores, batch_classes, batch_pred_bboxes

    def snap_tx_ty_tw_th_to_x1_y1_x2_y2(self, reg_heads, anchors):
        """
        sanp reg heads to pred bboxes
        reg_heads: [anchor_num, 4]   tx,ty,tw,th
        anchors: [anchor_num, 4]     x_min,y_min,x_max,y_max
        """
        device = reg_heads.device
        anchors = anchors.to(device)
        if reg_heads.shape[1] != 4 and anchors.shape[1] != 4:
            raise ValueError('shape expected anchor_num,4, but got {}'.format(reg_heads.shape))
        if reg_heads.shape[0] != anchors.shape[0]:
            raise ValueError('the number of reg_heads not equal anchors')

        anchors_wh = anchors[:, 2:] - anchors[:, :2]
        anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh

        factor = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device)
        reg_heads = reg_heads * factor
        pred_bboxes_wh = torch.exp(reg_heads[:, 2:]).to(device) * anchors_wh.to(device)
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
        ciou_methos = IoUMethod(iou_type="CIoU", box_type='xyxy')
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
                top1_class = single_classes[0]
                top1_pred_box = single_pred_bboxes[0]

                # single_keep_scores.append(top1_score)
                # single_keep_classes.append(detected_class)
                # single_keep_pred_bboxes.append(top1_pred_box)
                keep_scores.append(top1_score)
                keep_classes.append(top1_class)
                keep_pred_bboxes.append(top1_pred_box)

                top1_areas = single_pred_bboxes_areas[0]

                if single_class_scores.numel() == 1:
                    break
                single_class_scores = single_class_scores[1:]
                single_classes = single_classes[1:]
                single_pred_bboxes = single_pred_bboxes[1:, :]  # [N, 4]

                cious = ciou_methos(single_pred_bboxes, top1_pred_box.reshape(-1, 4))
                cious = cious.flatten()

                hidden_mask = cious < self.nms_threshold
                single_class_scores = single_class_scores[hidden_mask]
                single_classes = single_classes[hidden_mask]
                single_pred_bboxes = single_pred_bboxes[hidden_mask]
            # keep_scores.append(single_keep_scores)
            # keep_classes.append(single_keep_classes)
            # keep_pred_bboxes.append(single_keep_pred_bboxes)

        keep_scores = torch.tensor(keep_scores)
        keep_classes = torch.tensor(keep_classes)
        keep_pred_bboxes = torch.cat(keep_pred_bboxes).reshape((-1, 4))

        return keep_scores, keep_classes, keep_pred_bboxes

    @staticmethod
    def get_diou(pred_bbox, annots):
        """
        :param pred_bbox: [N, 4]
        :param annots: [M, 4]
        :return: [N, M]
        """
        pred_bbox_wh = pred_bbox[:, 2:] - pred_bbox[:, :2]
        pred_bbox_ctr = pred_bbox[:, :2] + 0.5 * pred_bbox_wh
        pred_bbox_areas = pred_bbox_wh[:, 0] * pred_bbox_wh[:, 1]  # [N]

        annots_wh = annots[:, 2:] - annots[:, :2]
        annots_ctr = annots[:, :2] + 0.5 * annots_wh
        annots_areas = annots_wh[:, 0] * annots_wh[:, 1]  # [M]

        res_dious = []
        for index, annot in enumerate(annots):
            overlap_x1 = torch.max(pred_bbox[:, 0], annot[0])
            overlap_y1 = torch.max(pred_bbox[:, 1], annot[1])
            overlap_x2 = torch.max(pred_bbox[:, 2], annot[2])
            overlap_y2 = torch.max(pred_bbox[:, 3], annot[3])
            overlaps = torch.clamp(overlap_x2 - overlap_x1, 0) * \
                       torch.clamp(overlap_y2 - overlap_y1, 0)
            unions = pred_bbox_areas + annots_areas - overlaps
            ious = overlaps / unions  # [N]

            # 计算对角线差距
            enclose_area_top_left = torch.min(annot[:2], pred_bbox[:, :2])  # [N, 2]
            enclose_area_bot_right = torch.max(annot[2:], pred_bbox[:, 2:])  # [N, 2]
            p2 = torch.pow(enclose_area_bot_right - enclose_area_top_left, 2)
            p2 = torch.sum(p2, dim=1)  # [N]

            # 计算中心点差距
            p1 = torch.pow(pred_bbox_ctr - annots_ctr, 2)  # [N， 2]
            p1 = torch.sum(p1, dim=1)  # [N]

            diou = (ious - p1 / p2).reshape(-1, 1)  # [N, 1]
            res_dious.append(diou)

        # res_cious sha[e [N, M]
        return torch.cat(res_dious, dim=1)


if __name__ == "__main__":
    per_image_preds = np.arange(200).reshape((10, 20))
    # per_image_cls_classes = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    #
    # sub1 = per_image_preds[:, per_image_cls_classes]
    # sub2 = per_image_preds[np.arange(10), per_image_cls_classes]
    # print(sub1, sub1.shape)
    # print(sub2, sub2.shape)
    # arr = np.array([[1, 2], [3, 4], [5, 6]])
    # print(arr.shape)
    # print(np.argmax(arr, axis=1))

    # arr1 = np.repeat(np.expand_dims(arr, 0), 2, axis=0)
    # arr2 = np.tile(arr, (2, 1, 1))
    # print(arr1, arr1.shape)
    # print(arr2, arr2.shape)
