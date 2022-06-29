import torch
from torch import Tensor, ThroughputBenchmark
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
    ares_bbs = (bboxs[:, 2] - bboxs[:, 0]) * (bboxs[:, 3] - bboxs[:, 1]) 
    area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])

    inter_0, inter_1 = torch.max(bboxs[:, 0], gt[0]).reshape(-1, 1), torch.max(bboxs[:, 1], gt[1]).reshape(-1, 1)
    inter_2, inter_3 = torch.min(bboxs[:, 2], gt[2]).reshape(-1, 1), torch.min(bboxs[:, 3], gt[3]).reshape(-1, 1)
    inters = torch.cat((inter_0, inter_1, inter_2, inter_3), dim=1)       # (N, 4)
    inters = torch.clamp(inters[:, 2] - inters[:, 0], min=0.) * \
             torch.clamp(inters[:, 3] - inters[:, 1], min=0.)             # (1, N)

    # print(inters)
    # print(ares_bbs, area_gt.item())
    unions = ares_bbs + area_gt - inters        # (1, N)
    ious = inters / unions      # (1, N)

    return ious


def calculate_tp(pred_boxes: Tensor, pred_scores: Tensor, gt_boxes: Tensor, gt_difficult: Tensor=None, iou_thresh: float = 0.5):
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
        # return 0, [], []   错误的返回, 漏掉了 FP
        # 如果gt为零, 但是还有某一类预测框存在的话, 这些框就属于FP, 即
        # 返回的 tp_list 中全为零, 个数为 len(pred_boxes)
        return 0, [0 for _ in range(pred_boxes)], pred_scores.flatten().to_list()

    # 若无对应的boxes，则 tp 为空
    if pred_boxes.numel() == 0:
        return len(gt_boxes), [], []

    # 否则计算所有预测框与gt之间的iou
    ious = pred_boxes.new_zeros((len(gt_boxes), len(pred_boxes)))   # 每一行代表所有bbox与某一个gt的iou
    for i in range(len(gt_boxes)):
        ious[i] = get_ious(pred_boxes, gt_boxes[i])

    # 在实际预测中，经常会出现多个预测框与同一个gt的IOU都大于阈值,
    # 这时通常只将这些预测框中 score 最大的算作TP，其它算作FP

    # max_ious, max_ious_idx = ious.max(dim=1)  # (M, N) -> (N, )
    # tp_lists = [0 for _ in range(len(pred_boxes))]
    # for iou_value, idx in zip(max_ious, max_ious_idx):
    #     if iou_value.item() > iou_thresh:
    #         tp_lists[idx] = 1
    
    tp_lists = [0 for _ in range(len(pred_boxes))]
    confidence_score = pred_scores.flatten()
    # 对每一行的iou分别进行计算, 找是否有匹配的gt
    for iou_list in ious:
        sub_iou_big_ids = np.where(iou_list>0.5)        # 返回的是 tuple, 元素个数和 ious 的维度相同
        sub_scores = confidence_score[sub_iou_big_ids[0]]
        sub_max_idx = np.argmax(sub_scores)
        idx = sub_iou_big_ids[0].tolist()[sub_max_idx]
        tp_lists[idx] = 1

    # print(confidence_score)
    # print(max_score, max_score_idx)
    # for iou_list in ious:
    #     print(iou_list, iou_list>iou_thresh)
        # idx = np.argmax(confidence_score[iou_list>iou_thresh])
        # print('idx = ', idx)
        # tp_lists[idx] = 1
    
    return len(gt_boxes), tp_lists, confidence_score


def calculate_pr(gt_num, tp_list, confidence_score):
    """
    计算Precision 和 Recall
    """
    if gt_num == 0:
        return [0], [0]
    if isinstance(confidence_score, (list, tuple)):
        confidence_score = np.array(confidence_score)

    if isinstance(tp_list, (list, tuple)):
        tp_list = np.array(tp_list)

    mask = np.argsort(-confidence_score)    # 按照score生成从大到小排序的mask
    tp_list = tp_list[mask]
    recall_list = [p/gt_num for p in np.cumsum(tp_list)]
    precision_list = [p/(idx+1) for idx, p in enumerate(np.cumsum(tp_list))]

    return precision_list, recall_list


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if isinstance(rec, (list, tuple)):
        rec = np.array(rec)

    if isinstance(prec, (list, tuple)):
        prec = np.array(prec)

    if use_07_metric:
        # 11 points metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(prec[rec >= t]) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        print(rec, prec)
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


if __name__ == "__main__":
    pred_bbs = torch.tensor([[10, 6, 15, 14], [5, 2, 10, 9], [7, 4, 12, 12], [17, 14, 20, 18],
                             [6, 14, 12, 18], [8, 9, 14, 15], [2, 20, 5, 25], [11, 7, 15, 16],
                             [8, 6, 13, 14], [10, 10, 14, 16], [10, 8, 14, 16]], dtype=torch.float32)
    # pred_ss = torch.rand(pred_bbs.shape[0], 1)
    pred_ss = torch.tensor([0.35084670782089233,
                            0.7216098308563232,
                            0.5988914370536804,
                            0.14170682430267334,
                            0.594139039516449,
                            0.80029363632202148,
                            0.9604676961898804,
                            0.7005996704101562,
                            0.9967637062072754,
                            0.5782052874565125,
                            0.7706285119056702]).reshape(-1, 1)

    gt_bbs = torch.tensor([[9, 8, 14, 15], [6, 5, 13, 11]], dtype=torch.float32)
    print(pred_bbs.shape, gt_bbs[0].shape)
    res = get_ious(pred_bbs, gt_bbs[1])
    print(res)

    # 需要注意, 这里不管传入多少个gt, 返回的 tp_list 都是 bbox 的个数
    # 只是其中可能有多个1, 1就代表tp
    # gt_num, tp_list, conf_scores = calculate_tp(pred_bbs, pred_ss, gt_bbs)
    # p_list, r_list = calculate_pr(gt_num, tp_list, conf_scores)
    # res = voc_ap(r_list, p_list)
    # print(res)
    # zipped = zip(tp_list, conf_scores)
    # zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
    # print(gt_num, tp_list, conf_scores)
    # for i in zipped:
    #     print(i)

    # a = np.array([[1, 2, 3], [4, 5, 6]])
    # res1 = np.cumsum(a)
    # res2 = np.cumsum(a, axis=0)
    # res3 = np.cumsum(a, axis=1)
    # print(res2)
    # print(res3)

    # scores = np.array(pred_ss.flatten())
    # ious = np.array([0.4706, 0.0145, 0.1905, 0.0000, 0.0536, 0.7317, 0.0000, 0.4200, 0.4706,
    #     0.5128, 0.7179])
    # sub_iou_big_ids = np.where(ious>0.5)        # 返回的是 tuple 
    # print(sub_iou_big_ids)
    # sub_scores = scores[sub_iou_big_ids]
    # print(sub_scores)
    # sub_ids = np.argmax(sub_scores)
    # print(sub_ids)

    # print(sub_iou_big_ids[0].tolist()[sub_ids])


