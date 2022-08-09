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
                        with torch.no_grad():
                            v = (4 / (math.pi ** 2)) * \
                                torch.pow(torch.arctan(bboxes2_wh[index2][0] / bboxes2_wh[index2][1]) -
                                          torch.arctan(bboxes1_wh[:, 0] / bboxes1_wh[:, 1]), 2)  # [N]
                            alpha = v / torch.clamp(1 - ious_src[index2] + v, min=1e-4)  # [N]
                        v = (4 / (math.pi ** 2)) * \
                            torch.pow(torch.arctan(bboxes2_wh[index2][0] / bboxes2_wh[index2][1]) -
                                      torch.arctan(bboxes1_wh[:, 0] / bboxes1_wh[:, 1]), 2)  # [N]
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


class IoUMethodMultiple:
    def __init__(self):
        pass

    def __call__(self, boxes1, boxes2, iou_type='IoU', box_type='xyxy'):
        """
        Args:
            boxes1: [I, N, 4]
            boxes2: [J, M, 4]
            iou_type:
            box_type:

        Returns:
            tensor()  shape: [max(I, J), max(M, N)]
            if boxes1 shape is [6, 9, 4]   boxes2 shape is [1, 6, 4]
            return shape is [6, 9]
        """
        assert iou_type in ['IoU', 'GIoU', 'DIoU', 'CIoU',
                            'EIoU'], 'wrong IoU type!'
        assert box_type in ['xyxy', 'xywh'], 'wrong box_type type!'

        if box_type == 'xywh':
            # transform format from [x_ctr,y_ctr,w,h] to xyxy
            boxes1_x1y1 = boxes1[..., 0:2] - boxes1[..., 2:4] / 2
            boxes1_x2y2 = boxes1[..., 0:2] + boxes1[..., 2:4] / 2
            boxes1 = torch.cat([boxes1_x1y1, boxes1_x2y2], dim=1)

            boxes2_x1y1 = boxes2[..., 0:2] - boxes2[..., 2:4] / 2
            boxes2_x2y2 = boxes2[..., 0:2] + boxes2[..., 2:4] / 2
            boxes2 = torch.cat([boxes2_x1y1, boxes2_x2y2], dim=1)

        overlap_area_xymin = torch.max(boxes1[..., 0:2], boxes2[..., 0:2])
        overlap_area_xymax = torch.min(boxes1[..., 2:4], boxes2[..., 2:4])
        overlap_area_sizes = torch.clamp(overlap_area_xymax -
                                         overlap_area_xymin,
                                         min=0)
        overlap_area = overlap_area_sizes[..., 0] * overlap_area_sizes[..., 1]

        boxes1_wh = torch.clamp(boxes1[..., 2:4] - boxes1[..., 0:2], min=0)
        boxes2_wh = torch.clamp(boxes2[..., 2:4] - boxes2[..., 0:2], min=0)

        boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
        boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

        # compute ious between boxes1 and boxes2
        union_area = boxes1_area + boxes2_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        ious = overlap_area / union_area

        if iou_type == 'IoU':
            return ious
        else:
            if iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
                enclose_area_top_left = torch.min(boxes1[..., 0:2],
                                                  boxes2[..., 0:2])
                enclose_area_bot_right = torch.max(boxes1[..., 2:4],
                                                   boxes2[..., 2:4])
                enclose_area_sizes = torch.clamp(enclose_area_bot_right -
                                                 enclose_area_top_left,
                                                 min=0)
                if iou_type in ['DIoU', 'CIoU', 'EIoU']:
                    # https://arxiv.org/abs/1911.08287v1
                    # compute DIoU c2 and p2
                    # c2:convex diagonal squared
                    c2 = enclose_area_sizes[...,
                                            0]**2 + enclose_area_sizes[...,
                                                                       1]**2
                    c2 = torch.clamp(c2, min=1e-4)
                    # p2:center distance squared
                    boxes1_ctr = (boxes1[..., 2:4] + boxes1[..., 0:2]) / 2
                    boxes2_ctr = (boxes2[..., 2:4] + boxes2[..., 0:2]) / 2
                    p2 = (boxes1_ctr[..., 0] - boxes2_ctr[..., 0])**2 + (
                        boxes1_ctr[..., 1] - boxes2_ctr[..., 1])**2
                    if iou_type == 'DIoU':
                        return ious - p2 / c2
                    elif iou_type == 'CIoU':
                        # compute CIoU v and alpha
                        v = (4 / math.pi**2) * torch.pow(
                            torch.atan(boxes2_wh[:, 0] / boxes2_wh[:, 1]) -
                            torch.atan(boxes1_wh[:, 0] / boxes1_wh[:, 1]), 2)

                        with torch.no_grad():
                            alpha = v / torch.clamp(1 - ious + v, min=1e-4)

                        return ious - (p2 / c2 + v * alpha)
                    elif iou_type == 'EIoU':
                        pw2 = (boxes2_wh[..., 0] - boxes1_wh[..., 0])**2
                        ph2 = (boxes2_wh[..., 1] - boxes1_wh[..., 1])**2
                        cw2 = enclose_area_sizes[..., 0]**2
                        ch2 = enclose_area_sizes[..., 1]**2
                        cw2 = torch.clamp(cw2, min=1e-4)
                        ch2 = torch.clamp(ch2, min=1e-4)

                        return ious - (p2 / c2 + pw2 / cw2 + ph2 / ch2)
                else:
                    enclose_area = enclose_area_sizes[:,
                                                      0] * enclose_area_sizes[:,
                                                                              1]
                    enclose_area = torch.clamp(enclose_area, min=1e-4)

                    return ious - (enclose_area - union_area) / enclose_area


if __name__ == "__main__":
    bb1s = torch.tensor([[10, 6, 15, 14], [5, 2, 10, 9], [7, 4, 12, 12], [17, 14, 20, 18],
                             [6, 10, 18, 18], [8, 9, 14, 15], [2, 20, 5, 25], [11, 7, 15, 16],
                             [8, 6, 13, 14], [10, 8, 23, 16], [10, 9, 24, 16]], dtype=torch.float32)
    bb2s = torch.tensor([[6., 5., 17., 11.], [9., 8., 19., 15.]])

    iou_method1 = IoUMethodMultiple()
    ious1 = iou_method1(bb1s.unsqueeze(1), bb2s.unsqueeze(0))

    iou_method2 = IoUMethod()
    ious2 = iou_method2(bb1s, bb2s)
    ious2 = ious2.transpose(1, 0)

    print(ious1, ious1.shape)
    print(ious2, ious2.shape)
    print(ious1 == ious2)
