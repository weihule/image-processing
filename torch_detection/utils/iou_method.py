import math
import numpy as np
import sys
import torch


class IoUMethod:
    def __init__(self, iou_type="IoU", box_type='xyxy'):
        assert iou_type in ['IoU', 'GIoU', 'CIoU', 'DIoU'], 'Wrong IoU Type'
        assert box_type in ['xyxy', 'xywh'], 'Wrong box_type'
        self.iou_type = iou_type
        self.box_type = 'xyxy'

    def __call__(self, bboxes1, bboxes2):
        """
        :param bboxes1: [N, 4]
        :param bboxes2: [M, 4]
        :return: [M, N]
        """
        if bboxes1.device != bboxes2.device:
            raise ValueError('two Tensor device should sample')
        res = torch.zeros((bboxes2.shape[0], bboxes1.shape[0])).to(bboxes2.device)
        if self.box_type == 'xywh':
            pass

        bboxes1_wh = torch.clamp(bboxes1[:, 2:] - bboxes1[:, :2], min=0)  # [N, 2]
        bboxes2_wh = torch.clamp(bboxes2[:, 2:] - bboxes2[:, :2], min=0)  # [M, 2]
        bboxes1_ctr = bboxes1[:, :2] + 0.5 * bboxes1_wh  # [N, 2]
        bboxes2_ctr = bboxes2[:, :2] + 0.5 * bboxes2_wh  # [M, 2]

        bboxes1_areas = bboxes1_wh[:, 0] * bboxes1_wh[:, 1]  # [N]
        bboxes2_areas = bboxes2_wh[:, 0] * bboxes2_wh[:, 1]  # [M]

        for index, bb2 in enumerate(bboxes2):
            overlap_xmin, overlap_ymin = torch.max(bboxes1[:, 0], bb2[0]), torch.max(bboxes1[:, 1], bb2[1])
            overlap_xmax, overlap_ymax = torch.min(bboxes1[:, 2], bb2[2]), torch.min(bboxes1[:, 3], bb2[3])
            overlap_size_w = torch.clamp(overlap_xmax - overlap_xmin, min=0)
            overlap_size_h = torch.clamp(overlap_ymax - overlap_ymin, min=0)
            overlap_areas = overlap_size_w * overlap_size_h

            unions = bboxes1_areas + torch.tile(bboxes2_areas[index], (len(overlap_areas),)) - overlap_areas
            ious = overlap_areas / unions
            res[index, :] = ious
        ious_src = res.clone()  # keep origin ious

        if self.iou_type == "IoU":
            return res

        if self.iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
            if self.iou_type in ['DIoU', 'CIoU']:
                for index2, bb2 in enumerate(bboxes2):
                    enclose_xmin, enclose_ymin = torch.min(bboxes1[:, 0], bb2[0]), torch.min(bboxes1[:, 1], bb2[1])
                    enclose_xmax, enclose_ymax = torch.max(bboxes1[:, 2], bb2[2]), torch.max(bboxes1[:, 3], bb2[3])

                    # compute p1 and p2
                    p2 = torch.pow(enclose_xmax - enclose_xmin, 2) + torch.pow(enclose_ymax - enclose_ymin, 2)
                    p1 = torch.pow(bboxes1_ctr[:, 0] - bboxes2_ctr[index2][0], 2) + \
                         torch.pow(bboxes1_ctr[:, 1] - bboxes2_ctr[index2][1], 2)
                    # p1, p2 = p1.to(bboxes2.device), p2.to(bboxes2.device)
                    res[index2] = res[index2] - p1 / p2
                    if self.iou_type == 'CIoU':
                        # compute v and alpha
                        v = (4 / (math.pi ** 2)) * \
                            torch.pow(torch.arctan(bboxes2_wh[index2][0] / bboxes2_wh[index2][1]) -
                                      torch.arctan(bboxes1_wh[:, 0] / bboxes1_wh[:, 1]), 2)  # [N]
                        with torch.no_grad():
                            alpha = v / torch.clamp(1 - ious_src[index2] + v, min=1e-4)  # [N]
                        res[index2] = res[index2] - alpha * v
                if self.iou_type == 'DIoU':
                    return res
                elif self.iou_type == 'CIoU':
                    return res


class IoUMethodNumpy:
    def __init__(self, iou_type="IoU", box_type='xyxy'):
        assert iou_type in ['IoU', 'GIoU', 'CIoU', 'DIoU'], 'Wrong IoU Type'
        assert box_type in ['xyxy', 'xywh'], 'Wrong box_type'
        self.iou_type = iou_type
        self.box_type = 'xyxy'

    def __call__(self, bboxes1, bboxes2):
        """
        :param bboxes1: [N, 4]
        :param bboxes2: [M, 4]
        :return: [M, N]
        """
        res = np.zeros((bboxes2.shape[0], bboxes1.shape[0]))
        if self.box_type == 'xywh':
            pass

        bboxes1_wh = np.clip(bboxes1[:, 2:] - bboxes1[:, :2], a_min=0, a_max=sys.maxsize)  # [N, 2]
        bboxes2_wh = np.clip(bboxes2[:, 2:] - bboxes2[:, :2], a_min=0, a_max=sys.maxsize)  # [M, 2]
        bboxes1_ctr = bboxes1[:, :2] + 0.5 * bboxes1_wh  # [N, 2]
        bboxes2_ctr = bboxes2[:, :2] + 0.5 * bboxes2_wh  # [M, 2]

        bboxes1_areas = bboxes1_wh[:, 0] * bboxes1_wh[:, 1]  # [N]
        bboxes2_areas = bboxes2_wh[:, 0] * bboxes2_wh[:, 1]  # [M]

        bboxes1_shape0 = bboxes1.shape[0]
        for index, bb2 in enumerate(bboxes2):
            overlap_xmin = np.maximum(bboxes1[:, 0], np.tile(bb2[0], bboxes1_shape0))
            overlap_ymin = np.maximum(bboxes1[:, 1], np.tile(bb2[1], bboxes1_shape0))
            overlap_xmax = np.minimum(bboxes1[:, 2], np.tile(bb2[2], bboxes1_shape0))
            overlap_ymax = np.minimum(bboxes1[:, 3], np.tile(bb2[3], bboxes1_shape0))
            overlap_size_w = np.clip(overlap_xmax - overlap_xmin, a_min=0, a_max=sys.maxsize)
            overlap_size_h = np.clip(overlap_ymax - overlap_ymin, a_min=0, a_max=sys.maxsize)
            overlap_areas = overlap_size_w * overlap_size_h

            unions = bboxes1_areas + np.tile(bboxes2_areas[index], (len(overlap_areas),)) - overlap_areas
            ious = overlap_areas / unions
            res[index, :] = ious
        ious_src = res.copy()  # keep origin ious

        if self.iou_type == "IoU":
            return res

        if self.iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
            if self.iou_type in ['DIoU', 'CIoU']:
                for index2, bb2 in enumerate(bboxes2):
                    bboxes1_shape0 = bboxes1.shape[0]
                    enclose_xmin = np.minimum(bboxes1[:, 0], np.tile(bb2[0], bboxes1_shape0))
                    enclose_ymin = np.minimum(bboxes1[:, 1], np.tile(bb2[1], bboxes1_shape0))
                    enclose_xmax = np.maximum(bboxes1[:, 2], np.tile(bb2[2], bboxes1_shape0))
                    enclose_ymax = np.maximum(bboxes1[:, 3], np.tile(bb2[3], bboxes1_shape0))

                    # compute p1 and p2
                    p2 = np.power(enclose_xmax - enclose_xmin, 2) + np.power(enclose_ymax - enclose_ymin, 2)
                    p1 = np.power(bboxes1_ctr[:, 0] - bboxes2_ctr[index2][0], 2) + \
                         np.power(bboxes1_ctr[:, 1] - bboxes2_ctr[index2][1], 2)
                    res[index2] = res[index2] - p1 / p2
                    if self.iou_type == 'CIoU':
                        # compute v and alpha
                        v = (4 / (math.pi ** 2)) * \
                            np.power(np.arctan(bboxes2_wh[index2][0] / bboxes2_wh[index2][1]) -
                                      np.arctan(bboxes1_wh[:, 0] / bboxes1_wh[:, 1]), 2)  # [N]
                        alpha = v / np.clip(1 - ious_src[index2] + v, a_min=1e-4, a_max=sys.maxsize)  # [N]
                        res[index2] = res[index2] - alpha * v
                if self.iou_type == 'DIoU':
                    return res
                elif self.iou_type == 'CIoU':
                    return res


if __name__ == "__main__":
    a = torch.tensor(66.)
    re_1 = torch.tile(a, (11,))
    print(a.shape)
    print(re_1)
