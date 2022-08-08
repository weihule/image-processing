import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from network_files.anchor import YoloV3Anchors

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from torch_detection.utils.iou_method import IoUMethod, IoUMethodMultiple


class YoloV4Loss(nn.Module):

    def __init__(self,
                 anchor_sizes=[[10, 13], [16, 30], [33, 23], [30, 61],
                               [62, 45], [59, 119], [116, 90], [156, 198],
                               [373, 326]],
                 strides=[8, 16, 32],
                 focal_eiou_gamma=0.5,
                 per_level_num_anchors=3,
                 conf_loss_weight=1.,
                 box_loss_weight=1.,
                 cls_loss_weight=1.,
                 box_loss_iou_type='CIoU',
                 iou_ignore_threshold=0.5):
        super(YoloV4Loss, self).__init__()
        assert box_loss_iou_type in [
            'IoU', 'GIoU', 'DIoU', 'CIoU', 'EIoU', 'Focal_EIoU'
        ], 'wrong IoU type!'

        self.anchors = YoloV3Anchors(
            anchor_sizes=anchor_sizes,
            strides=strides,
            per_level_num_anchors=per_level_num_anchors)
        self.anchor_sizes = anchor_sizes
        self.strides = strides
        self.focal_eiou_gamma = focal_eiou_gamma
        self.per_level_num_anchors = per_level_num_anchors
        self.conf_loss_weight = conf_loss_weight
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_iou_type = box_loss_iou_type
        self.iou_ignore_threshold = iou_ignore_threshold
        self.iou_function = IoUMethodSimple2Simple()

    def forward(self, preds, annotations):
        '''
        compute obj loss, reg loss and cls loss in one batch
        '''
        device = annotations.device
        batch_size = annotations.shape[0]
        obj_reg_cls_preds = preds[0]

        feature_size = [[
            per_level_cls_pred.shape[2], per_level_cls_pred.shape[1]
        ] for per_level_cls_pred in obj_reg_cls_preds]
        one_image_anchors = self.anchors(feature_size)
        batch_anchors = [
            torch.tensor(per_level_anchor).unsqueeze(0).repeat(
                batch_size, 1, 1, 1, 1).to(device)
            for per_level_anchor in one_image_anchors
        ]

        all_preds, all_targets = self.get_batch_anchors_targets(
            obj_reg_cls_preds, batch_anchors, annotations)

        # all_preds shape:[batch_size,anchor_nums,85]
        # reg_preds format:[scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax]
        # all_targets shape:[batch_size,anchor_nums,7]
        # targets format:[obj_target,box_loss_scale,x_offset,y_offset,scaled_gt_w,scaled_gt_h,class_target]

        conf_loss, reg_loss, cls_loss = self.compute_batch_loss(
            all_preds, all_targets)

        conf_loss = self.conf_loss_weight * conf_loss
        reg_loss = self.box_loss_weight * reg_loss
        cls_loss = self.cls_loss_weight * cls_loss

        loss_dict = {
            'conf_loss': conf_loss,
            'reg_loss': reg_loss,
            'cls_loss': cls_loss,
        }

        return loss_dict

    def compute_batch_loss(self, all_preds, all_targets):
        '''
        compute batch loss,include conf loss(obj and noobj loss,bce loss)、reg loss(CIoU loss)、cls loss(bce loss)
        all_preds:[batch_size,anchor_nums,85]
        all_targets:[batch_size,anchor_nums,8]
        '''
        device = all_targets.device
        all_preds = all_preds.view(-1, all_preds.shape[-1])
        all_targets = all_targets.view(-1, all_targets.shape[-1])

        positive_anchors_num = all_targets[all_targets[:, 7] > 0].shape[0]
        if positive_anchors_num == 0:
            return torch.tensor(0.).to(device), torch.tensor(0.).to(
                device), torch.tensor(0.).to(device)

        conf_preds = all_preds[:, 0:1]
        conf_targets = all_targets[:, 0:1]
        reg_preds = all_preds[all_targets[:, 0] > 0][:, 1:5]
        reg_targets = all_targets[all_targets[:, 0] > 0][:, 2:7]
        cls_preds = all_preds[all_targets[:, 0] > 0][:, 5:]
        cls_targets = all_targets[all_targets[:, 0] > 0][:, 7]

        # compute conf loss(obj and noobj loss)
        conf_preds = torch.clamp(conf_preds, min=1e-4, max=1. - 1e-4)
        temp_loss = -(conf_targets * torch.log(conf_preds) +
                      (1. - conf_targets) * torch.log(1. - conf_preds))
        obj_mask, noobj_mask = all_targets[:, 0:1], all_targets[:, 1:2]
        obj_sample_num = all_targets[all_targets[:, 0] > 0].shape[0]
        obj_loss = (temp_loss * obj_mask).sum() / obj_sample_num
        noobj_sample_num = all_targets[all_targets[:, 1] > 0].shape[0]
        noobj_loss = (temp_loss * noobj_mask).sum() / noobj_sample_num
        conf_loss = obj_loss + noobj_loss

        # compute reg loss
        box_loss_iou_type = 'EIoU' if self.box_loss_iou_type == 'Focal_EIoU' else self.box_loss_iou_type
        ious = self.iou_function(reg_preds,
                                 reg_targets[:, 1:5],
                                 iou_type=box_loss_iou_type,
                                 box_type='xyxy')
        reg_loss = (1 - ious) * reg_targets[:, 0]
        if self.box_loss_iou_type == 'Focal_EIoU':
            gamma_ious = self.iou_function(reg_preds,
                                           reg_targets[:, 1:5],
                                           iou_type='IoU',
                                           box_type='xyxy')
            gamma_ious = torch.pow(gamma_ious, self.focal_eiou_gamma)
            reg_loss = gamma_ious * reg_loss
        reg_loss = reg_loss.mean()

        # compute cls loss
        cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)
        cls_ground_truth = F.one_hot(cls_targets.long(),
                                     num_classes=cls_preds.shape[1] + 1)
        cls_ground_truth = (cls_ground_truth[:, 1:]).float()
        cls_loss = -(cls_ground_truth * torch.log(cls_preds) +
                     (1. - cls_ground_truth) * torch.log(1. - cls_preds))
        cls_loss = cls_loss.mean()

        return conf_loss, reg_loss, cls_loss

    def get_batch_anchors_targets(self, obj_reg_cls_heads, batch_anchors,
                                  annotations):
        '''
        Assign a ground truth target for each anchor
        '''
        device = annotations.device

        anchor_sizes = torch.tensor(self.anchor_sizes).float().to(device)
        anchor_sizes = anchor_sizes.view(
            len(anchor_sizes) // self.per_level_num_anchors,
            self.per_level_num_anchors, 2)
        # scale anchor size
        for i in range(anchor_sizes.shape[0]):
            anchor_sizes[i, :, :] = anchor_sizes[i, :, :] / self.strides[i]
        anchor_sizes = anchor_sizes.view(-1, 2)

        all_strides = [
            stride for stride in self.strides
            for _ in range(self.per_level_num_anchors)
        ]
        all_strides = torch.tensor(all_strides).float().to(device)

        grid_inside_ids = [
            i for _ in range(len(batch_anchors))
            for i in range(self.per_level_num_anchors)
        ]
        grid_inside_ids = torch.tensor(grid_inside_ids).to(device)

        all_preds,all_anchors, all_targets,feature_hw, per_layer_prefix_ids =[],[],[],[], [0, 0, 0]
        for layer_idx, (per_level_heads, per_level_anchors) in enumerate(
                zip(obj_reg_cls_heads, batch_anchors)):
            B, H, W, _, _ = per_level_anchors.shape
            for _ in range(self.per_level_num_anchors):
                feature_hw.append([H, W])
            if layer_idx == 0:
                for _ in range(self.per_level_num_anchors):
                    per_layer_prefix_ids.append(H * W *
                                                self.per_level_num_anchors)
                previous_layer_prefix = H * W * self.per_level_num_anchors
            elif layer_idx < len(batch_anchors) - 1:
                for _ in range(self.per_level_num_anchors):
                    cur_layer_prefix = H * W * self.per_level_num_anchors
                    per_layer_prefix_ids.append(previous_layer_prefix +
                                                cur_layer_prefix)
                previous_layer_prefix = previous_layer_prefix + cur_layer_prefix

            # obj target init value=0
            per_level_obj_target = torch.zeros(
                [B, H * W * self.per_level_num_anchors, 1],
                dtype=torch.float32,
                device=device)
            # noobj target init value=1
            per_level_noobj_target = torch.ones(
                [B, H * W * self.per_level_num_anchors, 1],
                dtype=torch.float32,
                device=device)
            # box loss scale init value=0
            per_level_box_loss_scale = torch.zeros(
                [B, H * W * self.per_level_num_anchors, 1],
                dtype=torch.float32,
                device=device)
            # reg target init value=0
            per_level_reg_target = torch.zeros(
                [B, H * W * self.per_level_num_anchors, 4],
                dtype=torch.float32,
                device=device)
            # cls target init value=-1
            per_level_cls_target = torch.ones(
                [B, H * W * self.per_level_num_anchors, 1],
                dtype=torch.float32,
                device=device) * (-1)
            # 8:[obj_target,noobj_target,box_loss_scale,scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax,class_target]
            per_level_targets = torch.cat([
                per_level_obj_target, per_level_noobj_target,
                per_level_box_loss_scale, per_level_reg_target,
                per_level_cls_target
            ],
                                          dim=-1)
            # per anchor format:[grids_x_index,grids_y_index,relative_anchor_w,relative_anchor_h,stride]
            per_level_anchors = per_level_anchors.view(
                per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

            per_level_heads = per_level_heads.view(per_level_heads.shape[0],
                                                   -1,
                                                   per_level_heads.shape[-1])
            per_level_obj_preds = per_level_heads[:, :, 0:1]
            per_level_cls_preds = per_level_heads[:, :, 5:]
            per_level_scaled_xy_ctr = per_level_heads[:, :, 1:
                                                      3] + per_level_anchors[:, :,
                                                                             0:
                                                                             2]
            per_level_scaled_wh = torch.exp(
                per_level_heads[:, :, 3:5]) * per_level_anchors[:, :, 2:4]
            per_level_scaled_xymin = per_level_scaled_xy_ctr - per_level_scaled_wh / 2
            per_level_scaled_xymax = per_level_scaled_xy_ctr + per_level_scaled_wh / 2
            # per reg preds format:[scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax]
            per_level_reg_preds = torch.cat(
                [per_level_scaled_xymin, per_level_scaled_xymax], dim=2)

            per_level_preds = torch.cat([
                per_level_obj_preds, per_level_reg_preds, per_level_cls_preds
            ],
                                        dim=2)

            all_preds.append(per_level_preds)
            all_anchors.append(per_level_anchors)
            all_targets.append(per_level_targets)

        all_preds = torch.cat(all_preds, dim=1)
        all_anchors = torch.cat(all_anchors, dim=1)
        all_targets = torch.cat(all_targets, dim=1)
        per_layer_prefix_ids = torch.tensor(per_layer_prefix_ids).to(device)
        feature_hw = torch.tensor(feature_hw).to(device)

        for img_idx, per_img_annots in enumerate(annotations):
            # drop all index=-1 class annotations
            per_img_annots = per_img_annots[per_img_annots[:, 4] >= 0]
            if per_img_annots.shape[0] != 0:
                # assert input annotations are[x_min,y_min,x_max,y_max,gt_class]
                # gt_class index range from 0 to 79
                gt_boxes = per_img_annots[:, 0:4]
                gt_classes = per_img_annots[:, 4]

                # for 9 anchors of each gt boxes,compute anchor global idx
                gt_9_boxes_ctr = (
                    (gt_boxes[:, 0:2] + gt_boxes[:, 2:4]) /
                    2).unsqueeze(1) / all_strides.unsqueeze(0).unsqueeze(-1)
                gt_9_boxes_grid_xy = torch.floor(gt_9_boxes_ctr)
                gt_9_boxes_grid_offset = gt_9_boxes_ctr - gt_9_boxes_grid_xy

                global_ids = ((gt_9_boxes_grid_xy[:, :, 1] *
                               feature_hw[:, 1].unsqueeze(0) +
                               gt_9_boxes_grid_xy[:, :, 0]) *
                              self.per_level_num_anchors +
                              grid_inside_ids.unsqueeze(0) +
                              per_layer_prefix_ids.unsqueeze(0)).long()

                # assign positive anchors which has max iou with a gt box
                # compute ious between 9 zero center gt bboxes and 9 zero center anchors
                gt_9_boxes_scaled_wh = (
                    gt_boxes[:, 2:4] - gt_boxes[:, 0:2]
                ).unsqueeze(1) / all_strides.unsqueeze(0).unsqueeze(-1)
                gt_9_boxes_xymin = -gt_9_boxes_scaled_wh / 2
                gt_9_boxes_xymax = gt_9_boxes_scaled_wh / 2
                gt_zero_ctr_9_boxes = torch.cat(
                    [gt_9_boxes_xymin, gt_9_boxes_xymax], dim=2)

                anchor_9_boxes_xymin = -anchor_sizes.unsqueeze(0) / 2
                anchor_9_boxes_xymax = anchor_sizes.unsqueeze(0) / 2
                anchor_zero_ctr_9_boxes = torch.cat(
                    [anchor_9_boxes_xymin, anchor_9_boxes_xymax], dim=2)

                positive_ious = self.iou_function(gt_zero_ctr_9_boxes,
                                                  anchor_zero_ctr_9_boxes,
                                                  iou_type='IoU',
                                                  box_type='xyxy')
                _, positive_anchor_idxs = positive_ious.max(axis=1)
                positive_anchor_idxs_mask = F.one_hot(
                    positive_anchor_idxs,
                    num_classes=anchor_sizes.shape[0]).bool()
                positive_global_ids = global_ids[
                    positive_anchor_idxs_mask].long()
                gt_9_boxes_scale = gt_9_boxes_scaled_wh / feature_hw.unsqueeze(
                    0)
                positive_gt_9_boxes_scale = gt_9_boxes_scale[
                    positive_anchor_idxs_mask]
                gt_9_scaled_boxes = gt_boxes.unsqueeze(
                    1) / all_strides.unsqueeze(0).unsqueeze(-1)
                positive_gt_9_scaled_boxes = gt_9_scaled_boxes[
                    positive_anchor_idxs_mask]

                # for positive anchor,assign obj target to 1(init value=0)
                all_targets[img_idx, positive_global_ids, 0] = 1
                # for positive anchor,assign noobj target to 0(init value=1)
                all_targets[img_idx, positive_global_ids, 1] = 0
                # for positive anchor,assign reg target:[box_loss_scale,scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax]
                all_targets[
                    img_idx, positive_global_ids,
                    2] = 2. - positive_gt_9_boxes_scale[:,
                                                        0] * positive_gt_9_boxes_scale[:,
                                                                                       1]
                all_targets[img_idx, positive_global_ids,
                            3:7] = positive_gt_9_scaled_boxes
                # for positive anchor,assign class target range from 1 to 80
                all_targets[img_idx, positive_global_ids, 7] = gt_classes + 1

                # assgin filter igonred anchors which ious>0.5 between anchor and gt boxes,set obj target value=-1(init=0,represent negative anchor)
                pred_scaled_bboxes = all_preds[img_idx:img_idx + 1, :, 1:5]
                gt_scaled_boxes = gt_boxes.unsqueeze(1) / all_anchors[
                    img_idx, :, 4:5].unsqueeze(0)
                filter_ious = self.iou_function(pred_scaled_bboxes,
                                                gt_scaled_boxes,
                                                iou_type='IoU',
                                                box_type='xyxy')
                filter_ious_max, _ = filter_ious.max(axis=0)
                # for ignored anchor,assign noobj target to 0(init value=1)
                all_targets[img_idx,
                            filter_ious_max > self.iou_ignore_threshold, 1] = 0

        return all_preds, all_targets


# # yolov4的anchor分配机制和yolov3一致, V5和前两个有所不同
# class YoloV4Loss(nn.Module):
#     def __init__(self,
#                  anchor_sizes=None,
#                  strides=None,
#                  per_level_num_anchors=3,
#                  conf_loss_weight=1.,
#                  box_loss_weight=1.,
#                  cls_loss_weight=1.,
#                  box_loss_iou_type='CIoU',
#                  iou_ignore_threshold=0.5):
#         super(YoloV4Loss, self).__init__()
#         assert box_loss_iou_type in ['IoU', 'DIoU', 'CIoU'], 'Wrong IoU type'
#         if anchor_sizes is None:
#             self.anchor_sizes = [[10, 13], [16, 30], [33, 23], [30, 61],
#                                  [62, 45], [59, 119], [116, 90], [156, 198],
#                                  [373, 326]]
#         if strides is None:
#             self.strides = [8, 16, 32]
#         self.anchors = YoloV3Anchors(
#             anchor_sizes=self.anchor_sizes,
#             strides=self.strides,
#             per_level_num_anchors=per_level_num_anchors
#         )
#         self.per_level_num_anchors = per_level_num_anchors
#         self.conf_loss_weight = conf_loss_weight
#         self.box_loss_weight = box_loss_weight
#         self.cls_loss_weight = cls_loss_weight
#         self.box_loss_iou_type = box_loss_iou_type
#         self.iou_ignore_threshold = iou_ignore_threshold
#         # self.iou_function = IoUMethod(iou_type=self.box_loss_iou_type)
#         self.iou_function = IoUMethodSimple2Simple()
#
#     def forward(self, preds, annotations):
#         """
#         compute obj loss, reg loss and cls loss in one batch
#         :param preds: [[B, H, W, 3, 85], ...]
#         :param annotations: [B, N, 5]
#         :return:
#         """
#         device = annotations.device
#         batch_size = annotations.shape[0]
#
#         # if input size:[B,3,416,416]
#         # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
#         # obj_reg_cls_heads shape:[[B, 52, 52, 3, 85],[B, 26, 26, 3, 85],[B, 13, 13, 3, 85]]
#
#         obj_reg_cls_preds = preds[0]
#
#         # feature_size = [[w, h], ...]
#         feature_size = [[per_level_cls_head.shape[2], per_level_cls_head.shape[1]] \
#                         for per_level_cls_head in obj_reg_cls_preds]
#         # one_image_anchors shape: [[52, 52, 3, 5], [26, 26, 3, 5], [13, 13, 3, 5]]
#         # 5: [grids_x_idx, grids_y_idx, relative_anchor_w, relative_anchor_h, stride] relative feature map
#         one_image_anchors = self.anchors(feature_size)
#
#         # batch_anchors shape is [[B, H, W, 3, 5], ...]
#         batch_anchors = [
#             torch.tensor(per_level_anchor).unsqueeze(0).repeat(
#                 batch_size, 1, 1, 1, 1).to(device) for per_level_anchor in one_image_anchors
#         ]
#
#         all_preds, all_targets = self.get_batch_anchors_targets(obj_reg_cls_preds,
#                                                                 batch_anchors,
#                                                                 annotations)
#
#         # all_preds shape:[batch_size,anchor_nums,85]
#         # reg_preds format:[scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax]
#         # all_targets shape:[batch_size,anchor_nums,8]
#         # targets format:[obj_target, noobj_target,
#         # box_loss_scale,x_offset,y_offset,scaled_gt_w,scaled_gt_h,class_target]
#
#         conf_loss, reg_loss, cls_loss = self.compute_batch_loss(
#             all_preds, all_targets)
#
#         conf_loss = self.conf_loss_weight * conf_loss
#         reg_loss = self.box_loss_weight * reg_loss
#         cls_loss = self.cls_loss_weight * cls_loss
#
#         loss_dict = {
#             'conf_loss': conf_loss,
#             'reg_loss': reg_loss,
#             'cls_loss': cls_loss,
#         }
#
#         return loss_dict
#
#     def get_batch_anchors_targets(self, obj_reg_cls_heads, batch_anchors, annotations):
#         """
#         Assign a ground truth target for each anchor
#         :param obj_reg_cls_heads: [[B, h, w, 3, 85], ...]
#         :param batch_anchors: [[B,52,52,3,5], [B,26,26,3,5], ...]
#                if one feature map shape is [w=3, h=5], this feature map anchor shape is [5, 3, 3, 5]
#         :param annotations: [B,N,5]
#         :return:
#             all_preds: [B, anchor_num, 85]
#             all_targets: [B, anchor_num, 8]
#         """
#         device = annotations.device
#
#         anchor_sizes = torch.tensor(self.anchor_sizes).float().to(device)
#         anchor_sizes = anchor_sizes.view(
#             len(anchor_sizes) // self.per_level_num_anchors, -1, 2)  # [3, 3, 2]
#         # scale anchor size
#         for i in range(anchor_sizes.shape[0]):
#             anchor_sizes[i] = anchor_sizes[i] / self.strides[i]
#         anchor_sizes = anchor_sizes.view(-1, 2)  # [9, 2]
#
#         # all_strides: [8, 8, 8, 16, 16, ...]
#         all_strides = [stride for stride in self.strides for _ in range(self.per_level_num_anchors)]
#         all_strides = torch.tensor(all_strides).to(device)
#
#         # grid_inside_ids: [0, 1, 2, 0, 1, 2, 0, 1, 2]
#         grid_inside_ids = [i for _ in range(len(batch_anchors)) for i in range(self.per_level_num_anchors)]
#         grid_inside_ids = torch.tensor(grid_inside_ids).to(device)
#
#         all_preds, all_anchors, all_targets = [], [], []
#         feature_hw = []  # 最终有9个元素, [[52,52], [52,52], ...]
#         per_layer_prefix_ids = [0, 0, 0]
#         # 分别遍历一个batch中所有图片的三个层级feature map
#         for layer_idx, (per_level_heads, per_level_anchors) in enumerate(zip(obj_reg_cls_heads, batch_anchors)):
#             # 这里 per_level_anchors 已经相对每个feature map做了缩小
#             B, H, W, _, _ = per_level_anchors.shape
#             for _ in range(self.per_level_num_anchors):
#                 feature_hw.append([H, W])
#
#             # TODO: 这里需要理解一下为什么
#             previous_layer_prefix, cur_layer_prefix = 0, 0
#             if layer_idx == 0:
#                 for _ in range(self.per_level_num_anchors):
#                     per_layer_prefix_ids.append(H * W * self.per_level_num_anchors)
#                 previous_layer_prefix = H * W * self.per_level_num_anchors
#             # len(batch_anchors) - 1 = 3-1 = 2
#             elif layer_idx < len(batch_anchors) - 1:
#                 for _ in range(self.per_level_num_anchors):
#                     cur_layer_prefix = H * W * self.per_level_num_anchors
#                     per_layer_prefix_ids.append(previous_layer_prefix + cur_layer_prefix)
#                 previous_layer_prefix = previous_layer_prefix + cur_layer_prefix
#
#             # obj target init value=0
#             per_level_obj_target = torch.zeros((B, H * W * self.per_level_num_anchors, 1),
#                                                dtype=torch.float32,
#                                                device=device)
#             # noobj target init value=1
#             per_level_noobj_target = torch.ones((B, H * W * self.per_level_num_anchors, 1),
#                                                 dtype=torch.float32,
#                                                 device=device)
#             # box loss scale init value=0
#             per_level_box_loss_scale = torch.zeros((B, H * W * self.per_level_num_anchors, 1),
#                                                    dtype=torch.float32,
#                                                    device=device)
#             # reg target init value=0
#             per_level_reg_target = torch.zeros((B, H * W * self.per_level_num_anchors, 4),
#                                                dtype=torch.float32,
#                                                device=device)
#             # cls target init value=-1
#             per_level_cls_target = torch.ones((B, H * W * self.per_level_num_anchors, 1),
#                                               dtype=torch.float32,
#                                               device=device) * (-1)
#             # per_level_targets shape is [B, H*W*self.per_level_num_anchors, 8]
#             # 8: [obj_target, noobj_target, box_loss_scale,
#             #       x_offset, y_offset, scaled_gt_w, scaled_gt_h, class_target]
#
#             per_level_targets = torch.cat((per_level_obj_target, per_level_noobj_target,
#                                            per_level_box_loss_scale, per_level_reg_target,
#                                            per_level_cls_target), dim=-1)
#
#             # per level anchor shape: [B, H*W*3, 5]
#             # 5: [grids_x_idx, grids_y_idx, relative_anchor_w, relative_anchor_h, stride]
#             per_level_anchors = per_level_anchors.view(
#                 per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])
#
#             # per_level_heads: [B, H*W*3, 85]
#             per_level_heads = per_level_heads.view(
#                 per_level_anchors.shape[0], -1, 85)
#             per_level_obj_preds = per_level_heads[..., 0:1]
#             per_level_cls_preds = per_level_heads[..., 5:]
#
#             # per_level_heads[..., 1:3] 是相对于某个cell的左上角偏移量, 所以加上per_level_anchors前两列得到中心点
#             # per_level_anchors这里anchors里的数值已经相对feature map做了缩小,在anchor.py中
#             # TODO: per_level_scaled_xy_ctr, per_level_scaled_wh 就是 bx, by, bw, bh
#             per_level_scaled_xy_ctr = per_level_heads[..., 1:3] + per_level_anchors[..., 0:2]  # [B, H*W*3, 2]
#             per_level_scaled_wh = torch.exp(per_level_heads[..., 3:5]) * per_level_anchors[..., 2:4]
#
#             per_level_scaled_xymin = per_level_scaled_xy_ctr - per_level_scaled_wh * 0.5
#             per_level_scaled_xymax = per_level_scaled_xy_ctr + per_level_scaled_wh * 0.5
#
#             # per_level_reg_heads shape: [B, H*W*3, 4]
#             # format:[scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax]
#             per_level_reg_heads = torch.cat((per_level_scaled_xymin, per_level_scaled_xymax), dim=2)
#
#             per_level_heads = torch.cat((per_level_obj_preds, per_level_reg_heads, per_level_cls_preds), dim=2)
#
#             all_preds.append(per_level_heads)  # [[B, H1*W1*3, 85], ...]
#             all_anchors.append(per_level_anchors)  # [[B, H1*W1*3, 5], ...]
#             all_targets.append(per_level_targets)  # [[B, H1*W1*3, 8], ...]
#
#         all_preds = torch.cat(all_preds, dim=1)  # [B, H1*W1*3+H2*W2*3+H3*W3*3, 85]
#         all_anchors = torch.cat(all_anchors, dim=1)  # [B, H1*W1*3+H2*W2*3+H3*W3*3, 5]
#         all_targets = torch.cat(all_targets, dim=1)  # [B, H1*W1*3+H2*W2*3+H3*W3*3, 8]
#         per_layer_prefix_ids = torch.tensor(per_layer_prefix_ids).to(device)
#         feature_hw = torch.tensor(feature_hw).to(device)
#
#         for img_idx, per_img_annots in enumerate(annotations):
#             # traverse each pic in batch
#             # drop all index=-1 in annotations  [N, 5]
#             one_image_annots = per_img_annots[per_img_annots[:, 4] >= 0]
#             # not empty
#             if one_image_annots.shape[0] > 0:
#                 # gt_class index range from 0 to 79
#                 gt_boxes = one_image_annots[:, :4]
#                 gt_classes = one_image_annots[:, 4]
#
#                 # for 9 anchor of each gt boxes, compute anchor global idx
#                 # gt_9_boxes_ctr: [gt_num, 2] -> [gt_num, 1, 2] -> [gt_num, 9, 2]
#                 # all_strides: [9, ] -> [1, 9] -> [1, 9, 1]
#                 # 这里9的意思是，有三个feature map,每个feature的每个cell有三个anchor,
#                 # 这三个anchor的左上坐标(cell的左上坐标)是一样的, 所以是9
#                 # 即前三个是一个feature map上的，中间三个是第二个feature map上的，后三个是第三个feature map上的
#                 gt_9_boxes_ctr = ((gt_boxes[:, :2] + gt_boxes[:, 2:]) * 0.5).unsqueeze(
#                     1) / all_strides.unsqueeze(0).unsqueeze(-1)
#
#                 # torch.floor向下取整到离它最近的整数
#                 # 如[[3.5, 2.1]] -> [[3, 2]], 即中心点为[3.5, 2.1]的gt是属于[3, 2]这个gird cell的
#                 gt_9_boxes_grid_xy = torch.floor(gt_9_boxes_ctr)
#                 # gt_9_boxes_grid_xy[:, :, 1] shape is [anchor_num, 9]
#
#                 # 这里应该是计算, gt映射在三个feature map上, 负责预测这些gt的anchor的索引
#                 # anchor总数是 (h1*w1*3 + ...)
#                 # global_ids shape [anchor_num, 9]
#                 # 含义是对于一个gt来说，映射在三个feature map上, 每一个feature map上的cell中有三个anchor对应去预测
#                 global_ids = ((gt_9_boxes_grid_xy[:, :, 1] * feature_hw[:, 1].unsqueeze(0) +
#                                gt_9_boxes_grid_xy[:, :, 0]) * self.per_level_num_anchors +
#                               grid_inside_ids.unsqueeze(0) + per_layer_prefix_ids.unsqueeze(0)).long()
#
#                 # assign positive anchor which has max iou with a gt box
#                 # [gt_num, 9, 2]
#                 gt_9_boxes_scaled_wh = (gt_boxes[:, 2:] - gt_boxes[:, :2]
#                                         ).unsqueeze(1) / all_strides.unsqueeze(0).unsqueeze(-1)
#                 # gt_9_boxes_xymin = gt_9_boxes_ctr - gt_9_boxes_scaled_wh * 0.5
#                 # gt_9_boxes_xymax = gt_9_boxes_ctr + gt_9_boxes_scaled_wh * 0.5
#                 gt_9_boxes_xymin = gt_9_boxes_scaled_wh * 0.5
#                 gt_9_boxes_xymax = gt_9_boxes_scaled_wh * 0.5
#
#                 # [gt_num, 9, 4]
#                 # 把这些anchor都移到以(0, 0)为中心的地方
#                 gt_zero_ctr_9_boxes = torch.cat((gt_9_boxes_xymin, gt_9_boxes_xymax), dim=2)
#
#                 # [1, 9, 2]
#                 anchor_9_boxes_xymin = -anchor_sizes.unsqueeze(0) * 0.5
#                 anchor_9_boxes_xymax = anchor_sizes.unsqueeze(0) * 0.5
#
#                 # [1, 9, 4]
#                 anchor_zero_ctr_9_boxes = torch.cat((anchor_9_boxes_xymin, anchor_9_boxes_xymax), dim=2)
#                 positive_ious = self.iou_function(gt_zero_ctr_9_boxes,
#                                                   anchor_zero_ctr_9_boxes,
#                                                   iou_type='IoU',
#                                                   box_type='xyxy')
#                 _, positive_anchor_idxs = positive_ious.max(dim=1)
#
#                 # positive_anchor_idxs_mask shape is [anchor_num, 9]
#                 positive_anchor_idxs_mask = F.one_hot(
#                     positive_anchor_idxs,
#                     num_classes=anchor_sizes.shape[0]).bool()
#                 # positive_global_ids shape is [anchor_num]
#                 positive_global_ids = global_ids[positive_anchor_idxs_mask].long()
#
#                 # gt_9_boxes_scale shape is [anchor_num, 9, 2]
#                 gt_9_boxes_scale = gt_9_boxes_scaled_wh / feature_hw.unsqueeze(0)
#                 # positive_gt_9_boxes_scale shape is [anchor_num, 2]
#                 positive_gt_9_boxes_scale = gt_9_boxes_scale[positive_anchor_idxs_mask]
#
#                 # gt_9_scaled_boxes shape is [anchor_num, 9, 4]
#                 gt_9_scaled_boxes = gt_boxes.unsqueeze(1) / all_strides.unsqueeze(0).unsqueeze(-1)
#
#                 # positive_gt_9_scaled_boxes shape is [anchor_num, 4]
#                 positive_gt_9_scaled_boxes = gt_9_scaled_boxes[positive_anchor_idxs_mask]
#
#                 # for positive anchor,assign obj target to 1(init value=0)
#                 all_targets[img_idx, positive_global_ids, 0] = 1
#                 # for positive anchor,assign noobj target to 0(init value=1)
#                 all_targets[img_idx, positive_global_ids, 1] = 0
#                 # for positive anchor,assign reg target:[box_loss_scale,scaled_xmin,scaled_ymin,scaled_xmax,scaled_ymax]
#                 all_targets[img_idx, positive_global_ids, 2] = \
#                     2. - positive_gt_9_boxes_scale[:, 0] * positive_gt_9_boxes_scale[:, 1]
#                 all_targets[img_idx, positive_global_ids, 3:7] = positive_gt_9_scaled_boxes
#                 # for positive anchor,assign class target range from 1 to 80
#                 all_targets[img_idx, positive_global_ids, 7] = gt_classes + 1
#
#                 # assign filter ignored anchors which ious>0.5
#                 # between anchor and gt boxes,set obj target value=-1(init=0,represent negative anchor)
#                 pred_scaled_bboxes = all_preds[img_idx:img_idx + 1, :, 1:5]
#                 gt_scaled_boxes = gt_boxes.unsqueeze(1) / all_anchors[img_idx, :, 4:5].unsqueeze(0)
#                 filter_ious = self.iou_function(pred_scaled_bboxes,
#                                                 gt_scaled_boxes,
#                                                 iou_type='IoU',
#                                                 box_type='xyxy')
#                 filter_ious_max, _ = filter_ious.max(axis=0)
#                 # for ignored anchor,assign noobj target to 0(init value=1)
#                 all_targets[img_idx, filter_ious_max > self.iou_ignore_threshold, 1] = 0
#
#         return all_preds, all_targets
#
#     def compute_batch_loss(self, all_preds, all_targets):
#         """
#         compute batch loss,include conf loss(obj and noobj loss,bce loss)、reg loss(CIoU loss)、cls loss(bce loss)
#         all_preds:[batch_size, anchor_nums, 85]
#         all_targets:[batch_size, anchor_nums, 8]
#         8 format: obj, noobj, box_loss_scale, x_offset, y_offset, scaled_gt_w, scaled_gt_h, class_target
#         """
#         device = all_targets.device
#         all_preds = all_preds.view(-1, all_preds.shape[-1])     # [B*anchor_nums, 85]
#         all_targets = all_targets.view(-1, all_targets.shape[-1])    # [B*anchor_nums, 8]
#
#         positive_anchors_num = all_targets[all_targets[:, 7] > 0].shape[0]
#         if positive_anchors_num == 0:
#             return torch.tensor(0.).to(device), torch.tensor(0.).to(
#                 device), torch.tensor(0.).to(device)
#
#         conf_preds = all_preds[:, 0:1]
#         conf_targets = all_targets[:, 0:1]
#
#         # all_preds的 reg部分已经变成左上和右下
#         reg_preds = all_preds[all_targets[:, 0] > 0][:, 1:5]
#         reg_targets = all_targets[all_targets[:, 0] > 0][:, 2:7]
#
#         cls_preds = all_preds[all_targets[:, 0] > 0][:, 5:]
#         cls_targets = all_targets[all_targets[:, 0] > 0][:, 7]
#
#         # compute conf loss(obj and noobj loss)
#         conf_preds = torch.clamp(conf_preds, min=1e-4, max=1. - 1e-4)
#         temp_loss = -(conf_targets * torch.log(conf_preds) +
#                       (1. - conf_targets) * torch.log(1. - conf_preds))
#         obj_mask, noobj_mask = all_targets[:, 0:1], all_targets[:, 1:2]
#         obj_sample_num = all_targets[all_targets[:, 0] > 0].shape[0]
#         obj_loss = (temp_loss * obj_mask).sum() / obj_sample_num
#         noobj_sample_num = all_targets[all_targets[:, 1] > 0].shape[0]
#         noobj_loss = (temp_loss * noobj_mask).sum() / noobj_sample_num
#         conf_loss = obj_loss + noobj_loss
#
#         # compute reg loss
#         box_loss_iou_type = 'EIoU' if self.box_loss_iou_type == 'Focal_EIoU' else self.box_loss_iou_type
#         ious = self.iou_function(reg_preds,
#                                  reg_targets[:, 1:5],
#                                  iou_type=box_loss_iou_type,
#                                  box_type='xyxy')
#         reg_loss = (1 - ious) * reg_targets[:, 0]
#         if self.box_loss_iou_type == 'Focal_EIoU':
#             gamma_ious = self.iou_function(reg_preds,
#                                            reg_targets[:, 1:5],
#                                            iou_type='IoU',
#                                            box_type='xyxy')
#             gamma_ious = torch.pow(gamma_ious, self.focal_eiou_gamma)
#             reg_loss = gamma_ious * reg_loss
#         reg_loss = reg_loss.mean()
#
#         # compute cls loss
#         cls_preds = torch.clamp(cls_preds, min=1e-4, max=1. - 1e-4)
#         num_classes = cls_preds.shape[1] + 1
#         cls_ground_truth = F.one_hot(cls_targets.long(),
#                                      num_classes=num_classes)
#         cls_ground_truth = (cls_ground_truth[:, 1:]).float()
#         cls_loss = -(cls_ground_truth * torch.log(cls_preds) +
#                      (1. - cls_ground_truth) * torch.log(1. - cls_preds))
#         cls_loss = cls_loss.mean()
#
#         return conf_loss, reg_loss, cls_loss
