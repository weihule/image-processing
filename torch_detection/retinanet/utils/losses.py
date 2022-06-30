import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import math
from PIL import Image


def snap_annotations_as_tx_ty_tw_th(anchors_gt_bboxes, anchors):
    """
    snap each anchor ground truth bbox form format:[x_min,y_min,x_max,y_max] to format:[tx,ty,tw,th]
    anchors_gt_bboxes: [M,4]
    anchors: [M,4]
    """
    if anchors_gt_bboxes.shape[0] != anchors.shape[0]:
        raise ValueError('anchors_gt_bboxes number not equal anchors number')
    anchors_w_h = anchors[:, 2:] - anchors[:, :2]  # [M, 2]
    anchors_center = anchors[:, :2] + 0.5 * anchors_w_h  # [M, 2]
    anchors_formed = torch.cat((anchors_center, anchors_w_h), dim=1)

    anchors_gt_bboxes_w_h = anchors_gt_bboxes[:, 2:] - anchors_gt_bboxes[:, :2]  # [M, 2]
    anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1.0)
    anchors_gt_bboxes_center = anchors_gt_bboxes[:, :2] + 0.5 * anchors_gt_bboxes_w_h  # [M, 2]
    anchors_gt_bboxes_formed = torch.cat((anchors_gt_bboxes_center, anchors_gt_bboxes_w_h), dim=1)

    snaped_annotations_for_anchors = torch.cat(
        [(anchors_gt_bboxes_center - anchors_center) / anchors_w_h,
         torch.log(anchors_gt_bboxes_w_h / anchors_w_h)], dim=1)

    # 另外需要说明的是,在许多faster rcnn的实现代码中,
    # 将box坐标按照faster rcnn中公式转换为tx，ty，tw，th后,
    # 这四个值又除以了[0.1,0.1,0.2,0.2]进一步放大
    factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
    # snaped_annotations_for_anchors = snaped_annotations_for_anchors / factor

    # snaped_annotations_for_anchors shape : [M, 4]
    return snaped_annotations_for_anchors


def compute_ious_for_one_image(one_image_anchors,
                               one_image_annotations):
    """
    compute ious between one image anchors and one image annotations
    """
    # make sure anchors format:[N,4],  [x_min,y_min,x_max,y_max]
    # make sure annotations format: [M,4],  [x_min,y_min,x_max,y_max]
    gt_num = one_image_annotations.shape[0]
    anchor_num = one_image_anchors.shape[0]
    res_iou = torch.zeros((gt_num, anchor_num), dtype=torch.float32)

    areas_anchors = (one_image_anchors[:, 2] - one_image_anchors[:, 0]) * \
                    (one_image_anchors[:, 3] - one_image_anchors[:, 1])  # torch.Size([N])
    areas_gts = (one_image_annotations[:, 2] - one_image_annotations[:, 0]) * \
                (one_image_annotations[:, 3] - one_image_annotations[:, 1])  # torch.Size([M])

    for idx, gt in enumerate(one_image_annotations):
        inters_0 = torch.max(one_image_anchors[:, 0], gt[0])
        inters_1 = torch.max(one_image_anchors[:, 1], gt[1])
        inters_2 = torch.min(one_image_anchors[:, 2], gt[2])
        inters_3 = torch.min(one_image_anchors[:, 3], gt[3])
        inters = torch.clamp(inters_2 - inters_0, min=0.) * \
                 torch.clamp(inters_3 - inters_1, min=0.)

        unions = areas_anchors + areas_gts[idx] - inters
        ious = inters / unions
        res_iou[idx, :] = ious

    # res_iou shape is [anchors_num, gts_num]
    return res_iou.transpose(0, 1)


def get_batch_anchors_annotations(batch_anchors, annotations):
    """
    Assign a ground truth box target and a ground truth class target for each anchor
    if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
    if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
    if anchor gt_class index > 0,this anchor is a object class anchor and used in calculate cls loss and reg loss
    batch_anchors: [B, N, 4]
    annotations: [B, M, 5]
    """
    device = annotations.device
    if batch_anchors.shape[0] != annotations.shape[0]:
        raise ValueError('batch number not equal')
    # 一个batch中每张图片含有的anchor数量
    one_image_anchor_nums = batch_anchors.shape[1]

    batch_anchors_annotations = list()
    # 开始处理每张图片的anchor
    for one_img_anchors, one_img_annots in zip(batch_anchors, annotations):
        # drop all index=-1 class annotations
        one_img_annots = one_img_annots[one_img_annots[:, 4] >= 0]  # [delta_M, 5] delta_M是已经去掉-1标签的gt的真实数量

        if one_img_annots.shape[0] == 0:
            # 如果该张图片中没有gt, 则该图片中的所有anchor都打上 -1 标签
            one_image_anchor_annots = torch.ones((one_image_anchor_nums, 5), device=device) * (-1)
        else:
            one_img_gt_bbs = one_img_annots[:, 0:4]  # gt的坐标部分      # [delta_M, 4]
            one_img_gt_cls = one_img_annots[:, 4]  # gt的标签部分        # [delta_M, 1]
            one_img_ious = compute_ious_for_one_image(one_img_anchors, one_img_gt_bbs)  # [anchors_num, gts_num]

            # snap per gt bboxes to the best iou anchor
            # 这里得到的是每一个anchor与该图片中所有gt的iou的最大值
            # 所以这也就可能导致多个anchor负责预测同一个gt
            overlap, indices = torch.max(one_img_ious, dim=1)
            # per_image_anchors_gt_bboxes就是每一个anchor对应预测的gt
            # 所以数量和图片中anchor的数量一致
            per_image_anchors_gt_bboxes = one_img_gt_bbs[indices]

            # per_image_anchors_gt_bboxes 和 one_img_anchors 的shape都是 [one_image_anchor_nums, 4]
            # transform gt bboxes to [tx,ty,tw,th] format for each anchor
            # one_image_anchors_snaped_boxes shape is [anchors_num, 4]
            one_image_anchors_snaped_boxes = snap_annotations_as_tx_ty_tw_th(per_image_anchors_gt_bboxes,
                                                                             one_img_anchors)

            one_image_anchors_gt_cls = torch.ones_like(overlap) * (-1)

            # if iou<0.4, assign anchors gt class as 0:background
            one_image_anchors_gt_cls[overlap < 0.4] = 0
            one_image_anchors_gt_cls[overlap > 0.5] = one_img_gt_cls[overlap > 0.5] + 1
            one_image_anchors_gt_cls = torch.unsqueeze(one_image_anchors_gt_cls, dim=1)  # [anchors_num, 1]

            # [anchors_num, 5]
            one_image_anchor_annots = torch.cat((one_image_anchors_snaped_boxes, one_image_anchors_gt_cls), dim=1)
        one_image_anchor_annots = torch.unsqueeze(one_image_anchor_annots, dim=0)
        batch_anchors_annotations.append(one_image_anchor_annots)
    batch_anchors_annotations = torch.cat(batch_anchors_annotations, dim=0)

    # batch anchors annotations shape:[batch_size, anchor_nums, 5]
    # 返回的是一个batch中所有图片的anchor信息和类别信息
    return batch_anchors_annotations


def custom_cross_entropy(input_data, target, num_class, use_custom=True):
    """
    :param use_custom: bool
    :param input_data: [N, num_class]
    :param target: [N]
    :param num_class: int
    :return:
    """
    if use_custom:
        one_hot = F.one_hot(target, num_classes=num_class).float()  # [N, num_class]
        custom_softmax = torch.exp(input_data) / torch.sum(torch.exp(input_data), dim=1).reshape((-1, 1))
        losses = -torch.sum(one_hot * torch.log(custom_softmax)) / input_data.shape[0]
    else:
        log_soft = F.log_softmax(input_data, dim=1)
        losses = F.nll_loss(log_soft, target)

    return losses


def custom_bce(input_data, target, use_custom=True):
    one_hot_target = F.one_hot(target, num_classes=2).float()

    if use_custom:
        losses = -one_hot_target * torch.log(torch.sigmoid(input_data)) \
                 - (1 - one_hot_target) * (torch.log(1 - torch.sigmoid(input_data)))
        # print(1-one_hot_target)
        # print(1-torch.log(input_data))
        # print(losses1, losses2)
        losses = torch.sum(losses) / (2 * input_data.shape[0])
    else:
        # losses = F.binary_cross_entropy(input_data, one_hot_target)
        losses = F.binary_cross_entropy_with_logits(input_data, one_hot_target)

    return losses


class RetinaLoss(nn.Module):
    def __init__(self,
                 image_w,
                 image_h,
                 alpha=0.25,
                 gamma=2,
                 beta=1.0 / 9.0,
                 epsilon=1e-4):
        super(RetinaLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.image_w = image_w
        self.image_h = image_h

    def compute_one_image_focal_loss(self, per_image_cls_heads,
                                     per_image_anchors_annotations):
        """
        compute one image focal loss(cls loss)
        per_image_cls_heads:[anchor_num,num_classes]
        per_image_anchors_annotations:[anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        per_image_cls_heads = per_image_cls_heads[per_image_anchors_annotations[:, 4] >= 0]
        per_image_anchors_annotations = per_image_anchors_annotations[per_image_anchors_annotations[:, 4] >= 0]
        per_image_cls_heads = torch.clamp(per_image_cls_heads,
                                          min=self.epsilon,
                                          max=1. - self.epsilon)
        num_classes = per_image_cls_heads.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(per_image_cls_heads[:, 4], num_classes=num_classes + 1).float()
        loss_ground_truth = loss_ground_truth[:, 1:]  # [anchor_num, num_classes]

        # [anchor_num, num_classes]
        alpha_factor = torch.ones_like(per_image_cls_heads) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.), alpha_factor, 1 - alpha_factor)

        # [anchor_num, num_classes]
        pt = torch.where(torch.eq(loss_ground_truth, 1.), per_image_cls_heads, 1 - per_image_cls_heads)
        focal_weight = alpha_factor * (1. - pt) ** self.gamma

        bce_loss = -loss_ground_truth * torch.log(per_image_cls_heads) \
                   - (1. - loss_ground_truth) * torch.log(1. - per_image_cls_heads)  # [anchor_num, num_classes]
        one_img_focal_loss = focal_weight * bce_loss
        one_img_focal_loss = one_img_focal_loss.sum()

        positive_anchor_num = per_image_anchors_annotations[per_image_anchors_annotations[:, 4] > 0].shape[0]

        return one_img_focal_loss / positive_anchor_num

    # 计算回归损失时, 只用正样本进行的loss计算
    def compute_one_image_smooth_l1_loss(self, per_image_reg_heads, per_image_anchors_annotations):
        """
        compute one image smoothl1 loss(reg loss)
        :param per_image_reg_heads: [anchor_num, 4]  B,4A,H,W -> B,H*W*A,4, 这里A取9, H*W*A也就是所有anchor的数量
        :param per_image_anchors_annotations: [anchor_num, 5]
        :return:
        """
        per_image_reg_heads = per_image_reg_heads[per_image_anchors_annotations[:, 4] > 0]
        per_image_anchors_annotations = per_image_anchors_annotations[per_image_anchors_annotations[:, 4] > 0]
        positive_anchor_num = per_image_anchors_annotations[per_image_anchors_annotations[:, 4] > 0].shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0.)

        # compute smoothl1 loss
        loss_gt = per_image_anchors_annotations[:, 0:4]
        x = torch.abs(loss_gt - per_image_reg_heads)    # [positive_anchor_num, 4]
        # 这里计算 smooth_l1_loss 时, 选用了beta值是 1/9, 相比原来的1, loss被放大
        # 因为原本是 小于1时, loss是0.5*x**2, 现在换成了 1/9,
        # 相当于之前 在（0.9-1）之间的数字都增大
        # [positive_anchor_num, 4]
        one_image_smooth_l1_loss = torch.where(torch.ge(x, self.beta), x - 0.5, (0.5*x**2) / self.beta)
        one_image_smooth_l1_loss = torch.sum(one_image_smooth_l1_loss, dim=1)
        one_image_smooth_l1_loss = one_image_smooth_l1_loss / positive_anchor_num   # [positive_anchor_num]

        return one_image_smooth_l1_loss

    def drop_out_border_anchors_and_heads(self):


if __name__ == "__main__":
    pred_bbs = torch.tensor([[100, 140, 120, 234], [5, 2, 10, 9], [7, 4, 12, 12], [17, 14, 20, 18],
                             [6, 14, 12, 18], [8, 9, 14, 15], [2, 20, 5, 25], [11, 7, 15, 16],
                             [8, 6, 13, 14], [10, 10, 14, 16], [10, 8, 14, 16], [18, 20, 40, 38]], dtype=torch.float32)
    gt_bbs = torch.tensor([[9, 8, 14, 15], [6, 5, 13, 11]], dtype=torch.float32)
    rl = RetinaLoss(32, 32)
    # res = snap_annotations_as_tx_ty_tw_th(gt_bbs, pred_bbs)
    # res = compute_ious_for_one_image(pred_bbs, gt_bbs)
    # print(res)

    from custome_dataset import VocDetection, Resizer, collater

    root = '/workshop/weihule/data/DL/VOCdataset'
    vd = VocDetection(root)
    # retina_resize = RetinaStyleResize()
    label_to_name = vd.voc_lable_to_categoty_id
    save_root = 'D:\\Desktop'

    # for i in range(0, 20, 4):
    #     batch_data = list()
    #     for j in range(i, i + 4):
    #         sample = vd[j]
    #         # res = RF(res)
    #         sample = Resizer()(sample)
    #         batch_data.append(sample)
    #     res = collater(batch_data)
    #     # print(res['img'].shape)
    #     # print(res['annot'])
    #     get_batch_anchors_annotations(pred_bbs.reshape(4, -1, 4), res['annot'])
    #     break

    # inputs = torch.rand(3, 5)
    # target = torch.tensor([0, 1, 4])
    # custom_loss1 = custom_cross_entropy(inputs, target, use_custom=True, num_class=5)
    # custom_loss2 = custom_cross_entropy(inputs, target, use_custom=False, num_class=5)
    # official_loss = F.cross_entropy(inputs, target)
    # print(custom_loss1, custom_loss2, official_loss)

    # inputs = torch.rand(4, 2)
    # inputs = torch.tensor([[0.8, 0.2]])
    # target = torch.tensor([0, 1, 1, 0])
    # cus_bce_loss = custom_bce(inputs, target)
    # official_loss = custom_bce(inputs, target, use_custom=False)
    # print(cus_bce_loss, official_loss)

    # x = torch.arange(30).reshape((10, 3))
    # y = torch.randint(low=0, high=9, size=(x.shape[0], 1))
    # arr = torch.cat((x, y), dim=1)
    # print(arr)
    # arr = arr[arr[:, 3] > 0]
    # print(arr)
