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
    anchors_gt_bboxes: [B, f1_anchor+f2_anchor..., 4], 是每一个anchor预测的gt, 是通过所有anchor与gt做iou取最大值得到的
    anchors: [B, f1_anchor+f2_anchor..., 4], 五个特征层输出anchor的拼接,
    这里求得是anchor相对gt的(tx,ty,tw,th),也是最终需要回归的数值
    """
    device = torch.device('cuda')
    anchors_gt_bboxes, anchors = anchors_gt_bboxes.to(device), anchors.to(device)
    if anchors_gt_bboxes.shape[0] != anchors.shape[0]:
        raise ValueError('anchors_gt_bboxes number not equal anchors number')
    anchors_w_h = anchors[:, 2:] - anchors[:, :2]  # [M, 2]
    anchors_center = anchors[:, :2] + 0.5 * anchors_w_h  # [M, 2]

    anchors_gt_bboxes_w_h = anchors_gt_bboxes[:, 2:] - anchors_gt_bboxes[:, :2]  # [M, 2]
    anchors_gt_bboxes_w_h = torch.clamp(anchors_gt_bboxes_w_h, min=1.0)
    anchors_gt_bboxes_center = anchors_gt_bboxes[:, :2] + 0.5 * anchors_gt_bboxes_w_h  # [M, 2]

    snaped_annotations_for_anchors = torch.cat(
        [(anchors_gt_bboxes_center - anchors_center) / anchors_w_h,
         torch.log(anchors_gt_bboxes_w_h / anchors_w_h)], dim=1)

    # 另外需要说明的是,在许多faster rcnn的实现代码中,
    # 将box坐标按照faster rcnn中公式转换为tx，ty，tw，th后,
    # 这四个值又除以了[0.1,0.1,0.2,0.2]进一步放大
    factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)
    snaped_annotations_for_anchors = snaped_annotations_for_anchors / factor

    # snaped_annotations_for_anchors shape : [M, 4]
    return snaped_annotations_for_anchors


def compute_ious_for_one_image(one_image_anchors,
                               one_image_annotations):
    """
    compute ious between one image anchors and one image annotations
    """
    # make sure anchors format:[N,4],  [x_min,y_min,x_max,y_max]
    # make sure annotations format: [M,4],  [x_min,y_min,x_max,y_max]
    device = one_image_annotations.device
    one_image_anchors = one_image_anchors.to(device)
    gt_num = one_image_annotations.shape[0]
    anchor_num = one_image_anchors.shape[0]
    res_iou = torch.zeros((gt_num, anchor_num), dtype=torch.float32)

    areas_anchors = (one_image_anchors[:, 2] - one_image_anchors[:, 0]) * \
                    (one_image_anchors[:, 3] - one_image_anchors[:, 1])  # torch.Size([N])
    areas_gts = (one_image_annotations[:, 2] - one_image_annotations[:, 0]) * \
                (one_image_annotations[:, 3] - one_image_annotations[:, 1])  # torch.Size([M])

    for idx, gt in enumerate(one_image_annotations):
        inters_0 = torch.max(one_image_anchors[:, 0], gt[0]).to(device)
        inters_1 = torch.max(one_image_anchors[:, 1], gt[1]).to(device)
        inters_2 = torch.min(one_image_anchors[:, 2], gt[2]).to(device)
        inters_3 = torch.min(one_image_anchors[:, 3], gt[3]).to(device)
        inters = torch.clamp(inters_2 - inters_0, min=0.) * \
                 torch.clamp(inters_3 - inters_1, min=0.)

        unions = torch.clamp(areas_anchors + areas_gts[idx] - inters, min=1e-4)
        ious = inters / unions
        res_iou[idx, :] = ious

    # res_iou shape is [anchors_num, gts_num]
    return res_iou.transpose(0, 1)


def get_batch_anchors_annotations(batch_anchors, annotations):
    """
    Assign a ground truth box target and a ground truth class target for each anchor
    if anchor gt_class index = -1,this anchor doesn't calculate cls loss and reg loss
    if anchor gt_class index = 0,this anchor is a background class anchor and used in calculate cls loss
    if anchor gt_class index > 0,this anchor is an object class anchor and used in calculate cls loss and reg loss
    batch_anchors: [B, N, 4]    N可能是63061, 五个特征层anchor数目的相加
    annotations: [B, M, 5]      M可能是14, 该batch图片中含有的最大gt数目
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
            one_image_anchor_annots = (torch.ones((one_image_anchor_nums, 5)) * (-1)).to(device)
        else:
            # print(one_img_annots, one_img_annots.shape)
            one_img_gt_bbs = one_img_annots[:, 0:4]  # gt的坐标部分      # [delta_M, 4]
            one_img_gt_cls = one_img_annots[:, 4]  # gt的标签部分        # [delta_M]
            one_img_ious = compute_ious_for_one_image(one_img_anchors, one_img_gt_bbs)  # [anchors_num, delta_M]

            # snap per gt bboxes to the best iou anchor
            # 这里得到的是每一个anchor与该图片中所有gt的iou的最大值
            # 所以这也就可能导致多个anchor负责预测同一个gt
            overlap, indices = torch.max(one_img_ious, dim=1)
            # print(f"len(overlap) = {len(overlap)}, len(indices) = {len(indices)}")
            # per_image_anchors_gt_bboxes就是每一个anchor对应预测的gt
            per_image_anchors_gt_bboxes = one_img_gt_bbs[indices]

            # per_image_anchors_gt_bboxes 和 one_img_anchors 的shape都是 [one_image_anchor_nums, 4]
            # transform gt bboxes to [tx,ty,tw,th] format for each anchor
            # one_image_anchors_snaped_boxes shape is [anchors_num, 4]
            one_image_anchors_snaped_boxes = snap_annotations_as_tx_ty_tw_th(per_image_anchors_gt_bboxes,
                                                                             one_img_anchors)

            # 这是对于上面one_image_anchors_snaped_boxes附标签
            one_image_anchors_gt_cls = (torch.ones_like(overlap) * (-1)).to(device)
            # if iou<0.4, assign anchors gt class as 0:background
            one_image_anchors_gt_cls[overlap < 0.4] = 0
            # a, b = one_image_anchors_gt_cls[overlap >= 0.5], indices[overlap >= 0.5] + 1
            one_image_anchors_gt_cls[overlap >= 0.5] = one_img_gt_cls[indices[overlap >= 0.5]] + 1
            one_image_anchors_gt_cls = torch.unsqueeze(one_image_anchors_gt_cls, dim=1)  # [anchors_num, 1]

            # [anchors_num, 5]
            one_image_anchor_annots = torch.cat((one_image_anchors_snaped_boxes, one_image_anchors_gt_cls), dim=1)
        one_image_anchor_annots = torch.unsqueeze(one_image_anchor_annots, dim=0)
        batch_anchors_annotations.append(one_image_anchor_annots)

    batch_anchors_annotations = torch.cat(batch_anchors_annotations, dim=0)

    # batch anchors annotations shape:[batch_size, anchor_nums, 5]
    # 返回的是一个batch中所有图片的anchor信息和类别信息, 也是最终做回归需要的信息
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


def drop_out_border_anchors_and_heads(cls_heads, reg_heads,
                                      batch_anchors, image_w, image_h):
    """
    dropout out of border anchors,cls heads and reg heads
    """
    # if cls_heads.shape[0] != reg_heads.shape[0] and reg_heads.shape[0] != batch_anchors.shape[0]:
    #     raise ValueError('batch number is different')
    # print(cls_heads.shape, reg_heads.shape, batch_anchors.shape)
    final_cls_heads, final_reg_heads, final_batch_anchors = list(), list(), list()
    for per_img_cls_head, per_img_reg_head, per_img_anchors in \
            zip(cls_heads, reg_heads, batch_anchors):
        left_top_x_mask = per_img_anchors[:, 0] > 0
        per_img_cls_head = per_img_cls_head[left_top_x_mask]
        per_img_reg_head = per_img_reg_head[left_top_x_mask]
        per_img_anchors = per_img_anchors[left_top_x_mask]

        left_top_y_mask = per_img_anchors[:, 1] > 0
        per_img_cls_head = per_img_cls_head[left_top_y_mask]
        per_img_reg_head = per_img_reg_head[left_top_y_mask]
        per_img_anchors = per_img_anchors[left_top_y_mask]

        right_down_x_mask = per_img_anchors[:, 2] < image_w
        per_img_cls_head = per_img_cls_head[right_down_x_mask]
        per_img_reg_head = per_img_reg_head[right_down_x_mask]
        per_img_anchors = per_img_anchors[right_down_x_mask]

        right_down_y_mask = per_img_anchors[:, 3] < image_h
        per_img_cls_head = per_img_cls_head[right_down_y_mask]
        per_img_reg_head = per_img_reg_head[right_down_y_mask]
        per_img_anchors = per_img_anchors[right_down_y_mask]

        final_cls_heads.append(torch.unsqueeze(per_img_cls_head, 0))
        final_reg_heads.append(torch.unsqueeze(per_img_reg_head, 0))
        final_batch_anchors.append(torch.unsqueeze(per_img_anchors, 0))

    final_cls_heads = torch.cat(final_cls_heads, 0)
    final_reg_heads = torch.cat(final_reg_heads, 0)
    final_batch_anchors = torch.cat(final_batch_anchors, 0)

    # 每个cell有A个anchor, 需要预测的是K个类别
    # [B, H*W*A-delta, K]
    # [B, H*W*A-delta, 4], 4指的是回归头输出的四个参数
    # [B, H*W*A-delta, 4], 4指的是anchor的左上右下四点坐标
    return final_cls_heads, final_reg_heads, final_batch_anchors


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

    def forward(self, cls_heads, reg_heads, batch_anchors, annotations):
        """
        compute cls loss ang reg loss in one batch
        :param cls_heads: list(), the output of model, example: [[B, 57600, 80], [B, 14400, 80], ...]
        :param reg_heads: list(), the output of model, example: [[B, 57600, 4], [B, 14400, 4], ...]
        :param batch_anchors: list(), the output of model, example: [[B, 57600, 4], [B, 14400, 4], ...]
        :param annotations: (N, 5) 5:[x_min, y_min, x_max, y_max, label]
        :return:
            cls_loss: torch.tensor(number)
            reg_loss: torch.tensor(number)
        """
        # 这里把每张图片5个特征层进行的合并
        cls_heads = torch.cat(cls_heads, dim=1)  # 由原来的list变成一个 [B, all_nums, 80]
        reg_heads = torch.cat(reg_heads, dim=1)
        batch_anchors = torch.cat(batch_anchors, dim=1)

        # import pdb
        # pdb.set_trace()
        cls_heads, reg_heads, batch_anchors = drop_out_border_anchors_and_heads(cls_heads, reg_heads,
                                                                                batch_anchors,
                                                                                self.image_w,
                                                                                self.image_h)
        # [anchor_num, 5]  5: tx, ty, tw, th, label
        batch_anchors_annotations = get_batch_anchors_annotations(batch_anchors, annotations)
        cls_loss, reg_loss = list(), list()
        valid_image_num = 0
        # 如果一张图片中没有 object, 那么anchor中就不会有正样本,
        # 就直接把这样图片的focal_loss 和 smooth_l1_loss 设为0.
        for per_img_cls_heads, per_img_reg_heads, per_img_anchors_annots, per_img_anchors in \
                zip(cls_heads, reg_heads, batch_anchors_annotations, batch_anchors):
            valid_anchors_num = (per_img_anchors_annots[per_img_anchors_annots[:, 4] > 0]).shape[0]
            if valid_anchors_num == 0:
                cls_loss.append(torch.tensor(0.))
                reg_loss.append(torch.tensor(0.))
            else:
                valid_image_num += 1
                one_img_cls_loss = self.compute_one_image_focal_loss(per_img_cls_heads, per_img_anchors_annots)
                # one_img_reg_loss = self.compute_one_image_smooth_l1_loss(per_img_reg_heads, per_img_anchors_annots)
                one_img_reg_loss = self.compute_one_image_ciou_loss(
                    per_image_reg_heads=per_img_reg_heads,
                    per_image_anchors_annotations=per_img_anchors_annots,
                    per_img_anchors=per_img_anchors
                )
                cls_loss.append(one_img_cls_loss)
                reg_loss.append(one_img_reg_loss)
        cls_loss = sum(cls_loss) / valid_image_num
        reg_loss = sum(reg_loss) / valid_image_num

        return cls_loss, reg_loss

    def compute_one_image_focal_loss(self, per_image_cls_heads,
                                     per_image_anchors_annotations):
        """
        compute one image focal loss(cls loss)
        per_image_cls_heads:[anchor_num,num_classes]
        per_image_anchors_annotations:[anchor_num,5]
        """
        # Filter anchors with gt class=-1, this part of anchor doesn't calculate focal loss
        # 在这里 per_image_anchors_annotations[:, 4] 为0的是负样本, 大于0的是正样本
        per_image_cls_heads = per_image_cls_heads[per_image_anchors_annotations[:, 4] >= 0]
        per_image_anchors_annotations = per_image_anchors_annotations[per_image_anchors_annotations[:, 4] >= 0]
        per_image_cls_heads = torch.clamp(per_image_cls_heads,
                                          min=self.epsilon,
                                          max=1. - self.epsilon)
        num_classes = per_image_cls_heads.shape[1]

        # generate 80 binary ground truth classes for each anchor
        loss_ground_truth = F.one_hot(per_image_anchors_annotations[:, 4].long(), num_classes=num_classes + 1)
        loss_ground_truth = loss_ground_truth.float()

        # 这块需要特别注意：
        # 正样本即在80个类别中某个类别的one hot向量值为1的样本，而负样本即在80个类别中所有类别的one hot向量均为0的样本。
        # 最后，根据RetinaNet论文中所述，由于使用了alpha和gamma，最后容易学习的负样本的loss值占总loss值的比例并不大，
        # 因此focal最后求和后只除以正样本的数量即可
        # [:, 1:]这个操作即让loss_ground_truth中负样本的one_hot编码全变成0
        loss_ground_truth = loss_ground_truth[:, 1:]  # [anchor_num, num_classes]

        # [anchor_num, num_classes]
        alpha_factor = torch.ones_like(per_image_cls_heads) * self.alpha
        alpha_factor = torch.where(torch.eq(loss_ground_truth, 1.), alpha_factor, 1 - alpha_factor)

        # [anchor_num, num_classes]
        pt = torch.where(torch.eq(loss_ground_truth, 1.), per_image_cls_heads, 1 - per_image_cls_heads)
        focal_weight = alpha_factor * torch.pow((1. - pt), self.gamma)

        bce_loss = -loss_ground_truth * torch.log(per_image_cls_heads) \
                   - (1. - loss_ground_truth) * torch.log(1. - per_image_cls_heads)  # [anchor_num, num_classes]
        one_img_focal_loss = focal_weight * bce_loss
        one_img_focal_loss = one_img_focal_loss.sum()

        # 是正样本喔, 所以需要剔除label为0的负样本才可以
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
        device = per_image_reg_heads.device
        per_image_reg_heads = per_image_reg_heads[per_image_anchors_annotations[:, 4] > 0]
        per_image_anchors_annotations = per_image_anchors_annotations[per_image_anchors_annotations[:, 4] > 0]
        positive_anchor_num = per_image_anchors_annotations.shape[0]
        # print(per_image_reg_heads.shape, per_image_anchors_annotations.shape)

        if positive_anchor_num == 0:
            return torch.tensor(0.).to(device)

        # compute smooth_l1 loss
        loss_gt = per_image_anchors_annotations[:, :4]
        x = torch.abs(loss_gt - per_image_reg_heads)  # [positive_anchor_num, 4]
        # 这里计算 smooth_l1_loss 时, 选用了beta值是 1/9, 相比原来的1, loss被放大
        # 因为原本是 小于1时, loss是0.5*x**2, 现在换成了 小于1/9, loss是0.5*x**2
        # 相当于之前 在（0.9-1）之间的数字都增大
        # [positive_anchor_num, 4]
        one_image_smooth_l1_loss = torch.where(torch.ge(x, self.beta),
                                               x - 0.5 * self.beta,
                                               0.5 * (x ** 2) / self.beta)
        # print(torch.mean(one_image_smooth_l1_loss, dim=1), one_image_smooth_l1_loss.shape)
        one_image_smooth_l1_loss = (torch.mean(one_image_smooth_l1_loss, dim=1)).sum()
        # one_image_smooth_l1_loss = one_image_smooth_l1_loss.sum()
        one_image_smooth_l1_loss = one_image_smooth_l1_loss / positive_anchor_num  # [positive_anchor_num]

        return one_image_smooth_l1_loss

    def compute_one_image_ciou_loss(self, per_image_reg_heads,
                                    per_image_anchors_annotations,
                                    per_img_anchors):
        """
        per_image_reg_heads: [anchor_num, 4]
        per_image_anchors_annotations: [anchor_num, 5]
        per_img_anchors: [anchor_num, 4]

        相对预测框和真实框而言的
        """
        device = per_image_reg_heads.device
        per_img_anchors = per_img_anchors.to(device)
        mask = per_image_anchors_annotations[:, 4] > 0
        per_image_reg_heads = per_image_reg_heads[mask]
        per_image_anchors_annotations = per_image_anchors_annotations[mask]
        per_img_anchors = per_img_anchors[mask]
        positive_anchor_num = per_image_reg_heads.shape[0]

        if positive_anchor_num == 0:
            return torch.tensor(0., device=device)

        per_image_anchor_w_h = per_img_anchors[:, 2:] - per_img_anchors[:, :2]
        per_image_anchor_ctr = per_img_anchors[:, :2] + 0.5 * per_image_anchor_w_h

        factor = torch.tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

        # 此处是解码, 将 reg_head 输出内容解码成预测框四点坐标
        # 1.解码 reg_heads
        per_image_reg_heads = per_image_reg_heads * factor
        pred_bbox_wh = torch.exp(per_image_reg_heads[:, 2:]) * per_image_anchor_w_h
        pred_bbox_ctr = per_image_reg_heads[:, :2] * per_image_anchor_w_h + per_image_anchor_ctr
        pred_bbox_x_min_y_min = pred_bbox_ctr - 0.5 * pred_bbox_wh
        pred_bbox_x_max_y_max = pred_bbox_ctr + 0.5 * pred_bbox_wh
        pred_bbox = torch.cat((pred_bbox_x_min_y_min, pred_bbox_x_max_y_max), dim=1)

        # 2. 解码 anchors_annotations
        # 上一步解码我们求得了所有的预测框坐标,第二部解码需要
        # 知道每一个预测框所对应的gt的坐标, 这个通过 anchors_annotations 解码
        per_image_anchors_annotations = per_image_anchors_annotations[:, :4] * factor
        annotations_wh = torch.exp(per_image_anchors_annotations[:, 2:]) * per_image_anchor_w_h
        annotations_ctr = per_image_anchors_annotations[:, :2] * per_image_anchor_w_h + per_image_anchor_ctr
        annotations_x_min_y_min = annotations_ctr - 0.5 * annotations_wh
        annotations_x_max_y_max = annotations_ctr + 0.5 * annotations_wh
        annotations_bbox = torch.cat((annotations_x_min_y_min, annotations_x_max_y_max), dim=1)

        # 计算预测框和真实框的iou, 这里计算每一个预测框和一个gt的iou,
        # 所以最后计算的iou的数量和预测框(或真实框)的数量一致
        overlap_top_left = torch.max(annotations_bbox[:, :2], pred_bbox[:, :2])
        overlap_top_right = torch.min(annotations_bbox[:, 2:], pred_bbox[:, 2:])
        overlap_area_sizes = torch.clamp(overlap_top_right - overlap_top_left, min=0)
        overlap_area = overlap_area_sizes[:, 0] * overlap_area_sizes[:, 1]

        pred_bbox_area = pred_bbox_wh[:, 0] * pred_bbox_wh[:, 1]
        annotation_bbox_area = annotations_wh[:, 0] * annotations_wh[:, 1]
        unions = pred_bbox_area + annotation_bbox_area - overlap_area
        ious = overlap_area / unions

        # 计算对角线差距
        enclose_area_top_left = torch.min(annotations_bbox[:, :2], pred_bbox[:, :2])
        enclose_area_bot_right = torch.max(annotations_bbox[:, 2:], pred_bbox[:, 2:])
        p2 = torch.pow(enclose_area_bot_right - enclose_area_top_left, 2)
        p2 = torch.sum(p2, dim=1)  # [anchor_num]

        # 计算中心点差距
        p1 = torch.pow(pred_bbox_ctr - annotations_ctr, 2)
        p1 = torch.sum(p1, dim=1)  # [anchor_num]

        # 注意alpha作为权重系数并不回传梯度
        with torch.no_grad():
            v = torch.pow(torch.atan(annotations_wh[:, 0] / annotations_wh[:, 1] -
                                     torch.atan(pred_bbox_wh[:, 0] / pred_bbox_wh[:, 1])), 2) * \
                4 / (math.e ** 2)
            alpha = v / (1 - ious + v)

        v = torch.pow(torch.atan(annotations_wh[:, 0] / annotations_wh[:, 1] -
                                 torch.atan(pred_bbox_wh[:, 0] / pred_bbox_wh[:, 1])), 2) * \
            4 / (math.e ** 2)
        ciou = ious - p1 / p2 - alpha * v
        ciou_loss = 1 - ciou
        ciou_loss = ciou_loss.sum() / positive_anchor_num
        ciou_loss = 2. * ciou_loss  # 这里乘2是为了平衡cls_loss

        return ciou_loss


if __name__ == "__main__":
    pred_bbs = torch.tensor([[100, 140, 120, 234], [5, 2, 10, 9], [7, 4, 12, 12], [17, 14, 20, 18],
                             [6, 14, 12, 18], [8, 9, 14, 15], [2, 20, 5, 25], [11, 7, 15, 16],
                             [8, 6, 13, 14], [10, 10, 14, 16], [10, 8, 14, 16], [18, 20, 40, 38]], dtype=torch.float32)
    gt_bbs = torch.tensor([[9, 8, 14, 15], [6, 5, 13, 11]], dtype=torch.float32)
    rl = RetinaLoss(32, 32)
    # res = snap_annotations_as_tx_ty_tw_th(gt_bbs, pred_bbs)
    res = compute_ious_for_one_image(pred_bbs, gt_bbs[0].reshape(-1, 4))
    res = res.flatten()
    print(res, res.shape)
    print(res < 0.05)

    from custom_dataset import VocDetection, Resizer, collater

    root = '/workshop/weihule/data/DL/VOCdataset'
    if not os.path.exists(root):
        root = '/nfs/home57/weihule/data/dl/VOCdataset'
    elif not os.path.exists(root):
        root = 'D:\\workspace\\data\\dl\\VOCdataset'
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

    # arr = torch.rand(3, 4)
    # indices = torch.tensor([0, 1, 1, 2, 1, 2])
    # print(arr)
    # print(arr[indices])
