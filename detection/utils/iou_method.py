import math
import numpy as np
import sys
import torch
import time

__all__ = [
    "IoUMethod",
    "IoUMethodNumpy",
    "IoUMethodNumpy2"
]


class IoUMethod:
    def __init__(self, iou_type="IoU", box_type='xyxy'):
        """
        xyxy type: [x_min, y_min, x_max. y_max]
        xywh type: [x_center, y_center, w, h]
        """
        assert iou_type in ['IoU', 'GIoU', 'CIoU', 'DIoU'], 'Wrong IoU Type'
        assert box_type in ['xyxy', 'xywh'], 'Wrong box_type'
        self.iou_type = iou_type
        self.box_type = box_type

    def __call__(self, boxes1, boxes2):
        """
        :param boxes1: [..., 4]
        :param boxes2: [..., 4]
        :return: [N, M]
        """
        print(f"in IoUMethod boxes1.shape={boxes1.shape}, boxes2.shape={boxes2.shape}")
        if boxes1.device != boxes2.device:
            raise ValueError('two Tensor device should sample')
        if self.box_type == 'xywh':
            # transform format from [x_ctr,y_ctr,w,h] to xyxy
            boxes1_x1y1 = boxes1[..., 0:2] - boxes1[..., 2:4] / 2
            boxes1_x2y2 = boxes1[..., 0:2] + boxes1[..., 2:4] / 2
            boxes1 = torch.cat([boxes1_x1y1, boxes1_x2y2], dim=1)

            boxes2_x1y1 = boxes2[..., 0:2] - boxes2[..., 2:4] / 2
            boxes2_x2y2 = boxes2[..., 0:2] + boxes2[..., 2:4] / 2
            boxes2 = torch.cat([boxes2_x1y1, boxes2_x2y2], dim=1)

        overlap_area_xymin = torch.max(boxes1[..., 0:2], boxes2[..., 0:2])
        overlap_area_xymax = torch.min(boxes1[..., 2:4], boxes2[..., 2:4])
        overlap_area_sizes = torch.clamp(overlap_area_xymax-overlap_area_xymin, min=0)
        # [boxes1.shape[0], boxes1.shape[1]]
        overlap_area = overlap_area_sizes[..., 0] * overlap_area_sizes[..., 1]

        boxes1_wh = torch.clamp(boxes1[..., 2:4]-boxes1[..., 0:2], min=0)
        boxes2_wh = torch.clamp(boxes2[..., 2:4]-boxes2[..., 0:2], min=0)

        boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
        boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

        union_area = boxes1_area + boxes2_area - overlap_area
        union_area = torch.clamp(union_area, min=1e-4)
        ious = overlap_area / union_area

        if self.iou_type == "IoU":
            return ious
        else:
            if self.iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
                enclose_area_top_left = torch.min(boxes1[..., 0:2], boxes2[..., 0:2])
                enclose_area_bot_right = torch.max(boxes1[..., 2:4], boxes2[..., 2:4])
                enclose_area_sizes = torch.clamp(enclose_area_bot_right-enclose_area_top_left,
                                                 min=0)
                if self.iou_type in ['DIoU', 'CIoU', 'EIoU']:
                    # 外接矩形对角线的距离的平方, 这里使用勾股定理算的
                    # 也可以直接使用两点距离来算
                    c2 = enclose_area_sizes[..., 0]**2 + enclose_area_sizes[..., 1]**2
                    c2 = torch.clamp(c2, min=1e-4)

                    # 两个框中心点的距离的平方
                    boxes1_ctr = (boxes1[..., 2:4] + boxes1[..., 0:2]) / 2
                    boxes2_ctr = (boxes2[..., 2:4] + boxes2[..., 0:2]) / 2
                    p2 = (boxes1_ctr[..., 0] - boxes2_ctr[..., 0])**2 + (boxes1_ctr[..., 1] - boxes2_ctr[..., 1])**2
                    if self.iou_type == "DIoU":
                        return ious - p2 / c2
                    elif self.iou_type == 'CIoU':
                        # compute CIoU v and alpha
                        print("===", boxes2_wh[:, 0].shape, boxes2_wh.shape)
                        v = (4 / math.pi**2) * torch.pow(
                            torch.atan(boxes2_wh[..., 0] / boxes2_wh[..., 1]) -
                            torch.atan(boxes1_wh[..., 0] / boxes1_wh[..., 1]), 2)

                        with torch.no_grad():
                            alpha = v / torch.clamp(1 - ious + v, min=1e-4)

                        return ious - (p2 / c2 + v * alpha)


class IoUMethodNumpy:
    def __init__(self, iou_type="IoU", box_type='xyxy'):
        assert iou_type in ['IoU', 'GIoU', 'CIoU', 'DIoU'], 'Wrong IoU Type'
        assert box_type in ['xyxy', 'xywh'], 'Wrong box_type'
        self.iou_type = iou_type
        self.box_type = box_type

    def __call__(self, boxes1, boxes2):
        """
        :param boxes1: [N, 4]
        :param boxes2: [M, 4]
        :return: [N, M]
        """
        res = np.zeros((boxes2.shape[0], boxes1.shape[0]))
        if self.box_type == 'xywh':
            # transform format from [x_ctr,y_ctr,w,h] to xyxy
            boxes1_x1y1 = boxes1[:, 0:2] - boxes1[:, 2:4] / 2
            boxes1_x2y2 = boxes1[:, 0:2] + boxes1[:, 2:4] / 2
            boxes1 = np.concatenate([boxes1_x1y1, boxes1_x2y2], axis=1)

            boxes2_x1y1 = boxes2[:, 0:2] - boxes2[:, 2:4] / 2
            boxes2_x2y2 = boxes2[:, 0:2] + boxes2[:, 2:4] / 2
            boxes2 = np.concatenate([boxes2_x1y1, boxes2_x2y2], axis=1)

        boxes1_wh = np.clip(boxes1[:, 2:] - boxes1[:, :2], a_min=0, a_max=sys.maxsize)  # [N, 2]
        boxes2_wh = np.clip(boxes2[:, 2:] - boxes2[:, :2], a_min=0, a_max=sys.maxsize)  # [M, 2]
        boxes1_ctr = boxes1[:, :2] + 0.5 * boxes1_wh  # [N, 2]
        boxes2_ctr = boxes2[:, :2] + 0.5 * boxes2_wh  # [M, 2]

        boxes1_areas = boxes1_wh[:, 0] * boxes1_wh[:, 1]  # [N]
        boxes2_areas = boxes2_wh[:, 0] * boxes2_wh[:, 1]  # [M]

        boxes1_shape0 = boxes1.shape[0]
        for index, bb2 in enumerate(boxes2):
            # bb2 shape is [4], 就是一个框的四点坐标
            overlap_xmin = np.maximum(boxes1[:, 0], np.tile(bb2[0], boxes1_shape0))
            overlap_ymin = np.maximum(boxes1[:, 1], np.tile(bb2[1], boxes1_shape0))
            overlap_xmax = np.minimum(boxes1[:, 2], np.tile(bb2[2], boxes1_shape0))
            overlap_ymax = np.minimum(boxes1[:, 3], np.tile(bb2[3], boxes1_shape0))
            overlap_size_w = np.clip(overlap_xmax - overlap_xmin, a_min=0, a_max=sys.maxsize)
            overlap_size_h = np.clip(overlap_ymax - overlap_ymin, a_min=0, a_max=sys.maxsize)
            overlap_areas = overlap_size_w * overlap_size_h

            unions = boxes1_areas + np.tile(boxes2_areas[index], boxes1_areas.shape) - overlap_areas
            ious = overlap_areas / unions
            res[index, :] = ious
        ious_src = res.copy()  # keep origin ious

        if self.iou_type == "IoU":
            return res

        if self.iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
            if self.iou_type in ['DIoU', 'CIoU']:
                for index2, bb2 in enumerate(boxes2):
                    boxes1_shape0 = boxes1.shape[0]
                    enclose_xmin = np.minimum(boxes1[:, 0], np.tile(bb2[0], boxes1_shape0))
                    enclose_ymin = np.minimum(boxes1[:, 1], np.tile(bb2[1], boxes1_shape0))
                    enclose_xmax = np.maximum(boxes1[:, 2], np.tile(bb2[2], boxes1_shape0))
                    enclose_ymax = np.maximum(boxes1[:, 3], np.tile(bb2[3], boxes1_shape0))

                    # compute p1 and p2
                    p2 = np.power(enclose_xmax - enclose_xmin, 2) + np.power(enclose_ymax - enclose_ymin, 2)
                    p1 = np.power(boxes1_ctr[:, 0] - boxes2_ctr[index2][0], 2) + \
                         np.power(boxes1_ctr[:, 1] - boxes2_ctr[index2][1], 2)
                    res[index2] = res[index2] - p1 / p2
                    if self.iou_type == 'CIoU':
                        # compute v and alpha
                        v = (4 / (math.pi ** 2)) * \
                            np.power(np.arctan(boxes2_wh[index2][0] / boxes2_wh[index2][1]) -
                                      np.arctan(boxes1_wh[:, 0] / boxes1_wh[:, 1]), 2)  # [N]
                        alpha = v / np.clip(1 - ious_src[index2] + v, a_min=1e-4, a_max=sys.maxsize)  # [N]
                        res[index2] = res[index2] - alpha * v
                if self.iou_type == 'DIoU':
                    return res
                elif self.iou_type == 'CIoU':
                    return res


class IoUMethodNumpy2:
    def __init__(self, iou_type="IoU", box_type='xyxy'):
        assert iou_type in ['IoU', 'GIoU', 'CIoU', 'DIoU'], 'Wrong IoU Type'
        assert box_type in ['xyxy', 'xywh'], 'Wrong box_type'
        self.iou_type = iou_type
        self.box_type = box_type

    def __call__(self, boxes1, boxes2):
        """
        :param boxes1: [..., 4]   11,1,4
        :param boxes2: [..., 4]   1,2,4
        :return:
        """
        res = np.zeros((boxes2.shape[0], boxes1.shape[0]))
        if self.box_type == 'xywh':
            # transform format from [x_ctr,y_ctr,w,h] to xyxy
            boxes1_x1y1 = boxes1[..., 0:2] - boxes1[..., 2:4] / 2
            boxes1_x2y2 = boxes1[..., 0:2] + boxes1[..., 2:4] / 2
            boxes1 = np.concatenate([boxes1_x1y1, boxes1_x2y2], axis=1)

            boxes2_x1y1 = boxes2[..., 0:2] - boxes2[..., 2:4] / 2
            boxes2_x2y2 = boxes2[..., 0:2] + boxes2[..., 2:4] / 2
            boxes2 = np.concatenate([boxes2_x1y1, boxes2_x2y2], axis=1)

        overlap_area_xymin = np.maximum(boxes1[..., 0:2], boxes2[..., 0:2])
        overlap_area_xymax = np.minimum(boxes1[..., 2:4], boxes2[..., 2:4])
        overlap_area_sizes = np.clip(overlap_area_xymax-overlap_area_xymin,
                                     a_min=0, a_max=sys.maxsize)
        overlap_area = overlap_area_sizes[..., 0] * overlap_area_sizes[..., 1]

        boxes1_wh = np.clip(boxes1[..., 2:4] - boxes1[..., 0:2], a_min=0, a_max=sys.maxsize)
        boxes2_wh = np.clip(boxes2[..., 2:4] - boxes2[..., 0:2], a_min=0, a_max=sys.maxsize)

        boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
        boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

        union_area = boxes1_area + boxes2_area - overlap_area
        union_area = np.clip(union_area, a_min=1e-4, a_max=sys.maxsize)
        ious = overlap_area / union_area

        if self.iou_type == "IoU":
            return ious
        else:
            if self.iou_type in ['GIoU', 'DIoU', 'CIoU', 'EIoU']:
                enclose_area_top_left = np.minimum(boxes1[..., 0:2], boxes2[..., 0:2])
                enclose_area_bot_right = np.maximum(boxes1[..., 2:4], boxes2[..., 2:4])
                enclose_area_sizes = np.clip(enclose_area_bot_right-enclose_area_top_left,
                                             a_min=0, a_max=sys.maxsize)
                if self.iou_type in ['DIoU', 'CIoU', 'EIoU']:
                    c2 = enclose_area_sizes[..., 0]**2 + enclose_area_sizes[..., 1]**2
                    c2 = np.clip(c2, a_min=0, a_max=sys.maxsize)

                    boxes1_ctr = boxes1[..., 0:2] + 0.5 * boxes1_wh
                    boxes2_ctr = boxes2[..., 0:2] + 0.5 * boxes2_wh
                    p2 = (boxes1_ctr[..., 0] - boxes2_ctr[..., 0])**2 + (
                        boxes1_ctr[..., 1] - boxes2_ctr[..., 1])**2
                    if self.iou_type == "DIoU":
                        return ious - p2 / c2


if __name__ == "__main__":
    bb1s = torch.tensor([[10, 6, 15, 14],
                         [5, 2, 10, 9],
                         [7, 4, 12, 12],
                         [17, 14, 20, 18],
                         [6, 10, 18, 18],
                         [8, 9, 14, 15],
                         [2, 20, 5, 25],
                         [11, 7, 15, 16],
                         [8, 6, 13, 14],
                         [10, 8, 23, 16],
                         [10, 9, 24, 16]], dtype=torch.float32)
    bb2s = torch.tensor([[6., 5., 17., 11.],
                         [9., 8., 19., 15.]])

    s1 = time.time()
    iou_method1 = IoUMethod(iou_type="CIoU")
    ious1 = iou_method1(bb1s.unsqueeze(1), bb2s.unsqueeze(0))
    print(ious1, ious1.shape)
    e1 = time.time()
    print(f"cost time = {e1-s1}")

    # iou_method2 = IoUMethodNumpy2(iou_type="DIoU")
    # bb1s = np.expand_dims(bb1s.numpy(), axis=1)
    # bb2s = np.expand_dims(bb2s.numpy(), axis=0)
    # ious2 = iou_method2(bb1s, bb2s)
    # # ious2 = ious2.transpose(1, 0)
    # print(ious2, ious2.shape)
    # print(f"cost time = {time.time() - e1}")

    iou_method3 = IoUMethod2()
    ious3 = iou_method3(bb1s.unsqueeze(1), bb2s.unsqueeze(0), iou_type="CIoU")
    print(ious3, ious3.shape)


