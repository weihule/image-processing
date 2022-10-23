import os
import sys
import collections
import numpy as np
import time
from tqdm import tqdm

import torch

sys.path.append(os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__))))
from util.avgmeter import AverageMeter


def compute_voc_ap(recall, precision, use_07_metric=False):
    if use_07_metric:
        # use voc 2007 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                # get max precision  for recall >= t
                p = np.max(precision[recall >= t])
            # average 11 recall point precision
            ap = ap + p / 11.
    else:
        # use voc>=2010 metric,average all different recall precision as ap
        # recall add first value 0. and last value 1.
        mrecall = np.concatenate(([0.], recall, [1.]))
        # precision add first value 0. and last value 0.
        mprecision = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mprecision.size - 1, 0, -1):
            mprecision[i - 1] = np.maximum(mprecision[i - 1], mprecision[i])

        # to calculate area under PR curve, look for points where X axis (recall) changes value
        i = np.where(mrecall[1:] != mrecall[:-1])[0]

        # sum (\Delta recall) * prec
        ap = np.sum((mrecall[i + 1] - mrecall[i]) * mprecision[i + 1])

    return ap


def compute_ious(a, b):
    """
    :param a: [N,(x1,y1,x2,y2)]
    :param b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """

    a = np.expand_dims(a, axis=1)  # [N,1,4]
    b = np.expand_dims(b, axis=0)  # [1,M,4]

    overlap = np.maximum(0.0,
                         np.minimum(a[..., 2:], b[..., 2:]) -
                         np.maximum(a[..., :2], b[..., :2]))  # [N,M,(w,h)]

    overlap = np.prod(overlap, axis=-1)  # [N,M]

    area_a = np.prod(a[..., 2:] - a[..., :2], axis=-1)
    area_b = np.prod(b[..., 2:] - b[..., :2], axis=-1)

    iou = overlap / (area_a + area_b - overlap)

    return iou


def evaluate_voc_detection(test_loader, model, criterion, decoder, args):
    # switch to evaluate mode
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    batch_size = args.test_batch

    with torch.no_grad():
        preds, gts = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for data in tqdm(test_loader):
            images, annots, scales, sizes = data['img'], data['annot'], data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            outs_tuple = model(images)

            loss_value = criterion(outs_tuple, annots)
            loss = sum(loss_value.values())
            losses.update(loss, images.size(0))

            pred_scores, pred_classes, pred_boxes = decoder(outs_tuple)

            pred_boxes /= np.expand_dims(np.expand_dims(scales, axis=-1), axis=-1)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.size(0))

            annots = annots.cpu().numpy()
            gt_bboxes, gt_classes = annots[:, :, 0:4], annots[:, :, 4]
            gt_bboxes /= np.expand_dims(np.expand_dims(scales, axis=-1), axis=-1)

            for per_image_pred_scores, per_image_pred_classes, per_image_pred_boxes, per_image_gt_bboxes, per_image_gt_classes, per_image_size in zip(
                    pred_scores, pred_classes, pred_boxes, gt_bboxes, gt_classes, sizes):
                per_image_pred_scores = per_image_pred_scores[per_image_pred_classes > -1]
                per_image_pred_boxes = per_image_pred_boxes[per_image_pred_classes > -1]
                per_image_pred_classes = per_image_pred_classes[per_image_pred_classes > -1]

                # clip boxes
                per_image_pred_boxes[:, 0] = np.maximum(per_image_pred_boxes[:, 0], 0)
                per_image_pred_boxes[:, 1] = np.maximum(per_image_pred_boxes[:, 1], 0)
                per_image_pred_boxes[:, 2] = np.minimum(per_image_pred_boxes[:, 2], per_image_size[1])
                per_image_pred_boxes[:, 3] = np.minimum(per_image_pred_boxes[:, 3], per_image_size[0])

                preds.append([
                    per_image_pred_boxes,
                    per_image_pred_classes,
                    per_image_pred_scores])

                per_image_gt_bboxes = per_image_gt_bboxes[per_image_gt_classes > -1]
                per_image_gt_classes = per_image_gt_classes[per_image_gt_classes > -1]

                gts.append([per_image_gt_bboxes, per_image_gt_classes])

            end = time.time()

        test_loss = losses.avg

        result_dict = collections.OrderedDict()

        # per image data load time(ms) and inference time(ms)
        per_image_load_time = data_time.avg / batch_size * 1000
        per_image_inference_time = batch_time.avg / batch_size * 1000

        result_dict['test_loss'] = test_loss
        result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
        result_dict['per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

        all_iou_threshold_map = collections.OrderedDict()
        all_iou_threshold_per_class_ap = collections.OrderedDict()
        for per_iou_threshold in tqdm(args.eval_voc_iou_threshold_list):
            per_iou_threshold_all_class_ap = collections.OrderedDict()
            for class_index in range(args.num_classes):
                per_class_gt_boxes = [image[0][image[1] == class_index] for image in gts]
                per_class_pred_boxes = [image[0][image[1] == class_index] for image in preds]
                per_class_pred_scores = [image[2][image[1] == class_index] for image in preds]

                fp = np.zeros((0,))
                tp = np.zeros((0,))
                scores = np.zeros((0,))
                total_gts = 0

                # loop for each sample
                for per_image_gt_boxes, per_image_pred_boxes, per_image_pred_scores in zip(
                        per_class_gt_boxes, per_class_pred_boxes, per_class_pred_scores):
                    total_gts = total_gts + len(per_image_gt_boxes)
                    # one gt can only be assigned to one predicted bbox
                    assigned_gt = []
                    # loop for each predicted bbox
                    for index in range(len(per_image_pred_boxes)):
                        scores = np.append(scores, per_image_pred_scores[index])
                        if per_image_gt_boxes.shape[0] == 0:
                            # if no gts found for the predicted bbox, assign the bbox to fp
                            fp = np.append(fp, 1)
                            tp = np.append(tp, 0)
                            continue
                        pred_box = np.expand_dims(per_image_pred_boxes[index], axis=0)
                        iou = compute_ious(per_image_gt_boxes, pred_box)
                        gt_for_box = np.argmax(iou, axis=0)
                        max_overlap = iou[gt_for_box, 0]
                        if max_overlap >= per_iou_threshold and gt_for_box not in assigned_gt:
                            fp = np.append(fp, 0)
                            tp = np.append(tp, 1)
                            assigned_gt.append(gt_for_box)
                        else:
                            fp = np.append(fp, 1)
                            tp = np.append(tp, 0)
                # sort by score
                indices = np.argsort(-scores)
                fp = fp[indices]
                tp = tp[indices]
                # compute cumulative false positives and true positives
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                # compute recall and precision
                recall = tp / total_gts
                precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                ap = compute_voc_ap(recall, precision, use_07_metric=False)
                per_iou_threshold_all_class_ap[class_index] = ap * 100

            per_iou_threshold_map = 0.
            for _, per_iou_threshold_per_class_ap in per_iou_threshold_all_class_ap.items():
                per_iou_threshold_map += float(per_iou_threshold_per_class_ap)
            per_iou_threshold_map /= args.num_classes
            all_iou_threshold_map[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,mAP'] = per_iou_threshold_map
            all_iou_threshold_per_class_ap[
                f'IoU={per_iou_threshold:.2f},area=all,maxDets=100,per_class_ap'] = per_iou_threshold_all_class_ap

        for key, value in all_iou_threshold_map.items():
            result_dict[key] = value
        for key, value in all_iou_threshold_per_class_ap.items():
            result_dict[key] = value

        return result_dict
