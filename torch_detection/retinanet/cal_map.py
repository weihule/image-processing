import torch
from torch import Tensor
import numpy as np
import time
import timeit
from shapely.geometry import Polygon


def get_iou(bbox1, bbox2):
    """
    计算单个框的iou
    """
    inter_0, inter_1 = max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1])
    inter_2, inter_3 = min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3])

    if inter_0 >= min(bbox1[2], bbox2[2]):
        return 0
    
    union = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    inter = (inter_2 - inter_0) * (inter_3 - inter_1)

    return float(inter / union)

    
    # inter = (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])) * \
    #         (min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))


def get_ious(bboxs, gt):

    union = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1]) +  (gt[2] - gt[0]) * (gt[3] - gt[1])


def calculate_tp(pred_boxes: Tensor, pred_scores: Tensor, gt_boxes: Tensor, gt_difficult: Tensor, iou_thresh: float = 0.5):
    """
        calculate tp/fp for all predicted bboxes for one class of one image.
        对于匹配到同一gt的不同bboxes, 让score最高tp = 1, 其它的tp = 0
    Args:
        pred_boxes: Tensor[N, 4], 某张图片中某类别的全部预测框的坐标 (x0, y0, x1, y1)
        pred_scores: Tensor[N, 1], 某张图片中某类别的全部预测框的score
        gt_boxes: Tensor[M, 4], 某张图片中某类别的全部gt的坐标 (x0, y0, x1, y1)
        gt_difficult: Tensor[M, 1], 某张图片中某类别的gt中是否为difficult目标的值
        iou_thresh: iou 阈值
    Returns:
        gt_num: 某张图片中某类别的gt数量
        tp_list: 记录某张图片中某类别的预测框是否为tp的情况
        confidence_score: 记录某张图片中某类别的预测框的score值 (与tp_list相对应)
    """
    if gt_boxes.numel() == 0:
        return 0, [], []

    # 若无对应的boxes，则 tp 为空
    if pred_boxes.numel() == 0:
        return len(gt_boxes), [], []

    # 否则计算所有预测框与gt之间的iou
    ious = pred_boxes.new_zeros((len(gt_boxes), len(pred_boxes)))   # 每一行代表所有bbox与某一个gt的iou
    
    


if __name__ == "__main__":
    start = time.clock()
    # arr1 = [1, 1, 4, 1, 4, 6, 1, 6]
    # arr2 = [2, 3, 7, 3, 7, 9, 2, 9]
    # area1 = Polygon(np.array(arr1).reshape(4, 2))
    # area2 = Polygon(np.array(arr2).reshape(4, 2))
    # iou = area1.intersection(area2).area / (area1.area + area2.area)

    # ar1 = [1, 1, 4, 6]
    ar1 = [10, 10, 14, 14]
    ar2 = [2, 3, 7, 9]
    iou = get_iou(ar1, ar2)

    print(f'{iou}, running time: {time.clock()-start}')


    pred_bbox = torch.randn(7, 4)
    print(pred_bbox, pred_bbox[:, 1])
    a1 = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
    b1 = torch.tensor(10)
    print(a1)
    print(a1+b1)
    # temp = pred_bbox.new_zeros(10, 10)
    # print(len(temp), temp.numel())

    features = np.array([[0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 0, 1],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
                [1, 0, 1, 2],
                [1, 0, 1, 2],
                [2, 0, 1, 2],
                [2, 0, 1, 1],
                [2, 1, 0, 1],
                [2, 1, 0, 2],
                [2, 0, 0, 0]])

    labels = np.array(['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes']).reshape((-1, 1))

    # dataset = np.hstack((features, labels))
    dataset = np.concatenate((features, labels), axis=1)
    mask = features[:, 2] == 0		# 根据索引为2的列生成mask
    subdataset = features[mask]		# 将mask为false的行全都去除
    print(subdataset)
    subdataset = np.delete(subdataset, 2, axis=1)	# 删除索引为2的列
