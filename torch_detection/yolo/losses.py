import os
import sys
import torch
import torch.nn as nn
from network_files.anchor import YoloV3Anchors

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(BASE_DIR)

from torch_detection.utils.iou_methos import IoUMethod


# yolov4的anchor分配机制和yolov3一致, V4和前两个有所不同
class YoloV4Loss(nn.Module):
    def __init__(self,
                 anchor_sizes=None,
                 strides=None,
                 per_level_num_anchors=3,
                 obj_layer_weight=None,
                 conf_loss_weight=1.,
                 box_loss_weight=1.,
                 cls_loss_weight=1.,
                 box_loss_iou_type='CIoU',
                 iou_ignore_threshold=0.5):
        super(YoloV4Loss, self).__init__()
        assert box_loss_iou_type in ['IoU, DIoU, CIoU'], 'Wrong IoU type'
        if anchor_sizes is None:
            self.anchor_sizes = [[10, 13], [16, 30], [33, 23], [30, 61],
                                 [62, 45], [59, 119], [116, 90], [156, 198],
                                 [373, 326]]
        if strides is None:
            self.strides = [8, 16, 32]
        if obj_layer_weight is None:
            self.obj_layer_weight = [4.0, 1.0, 0.4]
        self.anchors = YoloV3Anchors(
            anchor_sizes=self.anchor_sizes,
            strides=self.strides,
            per_level_num_anchors=per_level_num_anchors
        )
        self.per_level_num_anchors = per_level_num_anchors
        self.conf_loss_weight = conf_loss_weight
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.box_loss_iou_type = box_loss_iou_type
        self.iou_ignore_threshold = iou_ignore_threshold
        self.iou_function = IoUMethod(iou_type=self.box_loss_iou_type)

    def forward(self, preds, annotations):
        """
        compute obj loss, reg loss and cls loss in one batch
        :param preds:
        :param annotations: [B, N, 5]
        :return:
        """
        device = annotations.device
        batch_size = annotations.shape[0]

        # if input size:[B,3,416,416]
        # features shape:[[B, 255, 52, 52],[B, 255, 26, 26],[B, 255, 13, 13]]
        # obj_reg_cls_heads shape:[[B, 52, 52, 3, 85],[B, 26, 26, 3, 85],[B, 13, 13, 3, 85]]
        # TODO: 这块需要查一下
        obj_reg_cls_preds = preds[0]

        # feature_size = [[w, h], ...]
        feature_size = [[per_level_cls_head[2], per_level_cls_head[1]] for per_level_cls_head in obj_reg_cls_preds]
        # one_image_anchors shape: [[52, 52, 3, 5], [26, 26, 3, 5], [13, 13, 3, 5]]
        # 5: [grids_x_idx, grids_y_idx, relative_anchor_w, relative_anchor_h, stride] relative feature map
        one_image_anchors = self.anchors(feature_size)

        # batch_anchors shape is [[B, H, W, 3, 5], ...]
        batch_anchors = [
            torch.tensor(per_level_anchor).unsqueeze(0).repeat(
                batch_size, 1, 1, 1, 1) for per_level_anchor in one_image_anchors
        ]

        all_anchors, all_targets = self.get_batch_anchors_targets(batch_anchors, annotations)

    def get_batch_anchors_targets(self, obj_reg_cls_heads, batch_anchors, annotations):
        """
        Assign a ground truth target for each anchor
        :param obj_reg_cls_heads: [[B, h, w, 3, 85], ...]
        :param batch_anchors: [[B,52,52,3,5], [B,26,26,3,5], ...]
               if one feature map shape is [w=3, h=5], this feature map anchor shape is [5, 3, 3, 5]
        :param annotations: [B,N,5]
        :return:
        """
        device = annotations.device

        anchor_sizes = torch.tensor(self.anchor_sizes).float().to(device)
        anchor_sizes = anchor_sizes.view(
            len(anchor_sizes) // self.per_level_num_anchors, -1, 2)  # [3, 3, 2]
        # scale anchor size
        for i in range(anchor_sizes.shape[0]):
            anchor_sizes[i] = anchor_sizes[i] / self.strides[i]
        anchor_sizes = anchor_sizes.view(-1, 2)  # [9, 2]

        # all_strides: [8, 8, 8, 16, 16, ...]
        all_strides = [stride for stride in self.strides for _ in range(self.per_level_num_anchors)]
        all_strides = torch.tensor(all_strides).to(device)

        # grid_inside_ids: [0, 1, 2, 0, 1, 2, 0, 1, 2]
        grid_inside_ids = [i for _ in range(len(batch_anchors)) for i in range(self.per_level_num_anchors)]
        grid_inside_ids = torch.tensor(grid_inside_ids).to(device)

        all_preds, all_anchors, all_targets = [], [], []
        feature_hw = []
        per_layer_prefix_ids = [0, 0, 0]
        # 分别遍历一个batch中所有图片的三个层级feature map
        for layer_idx, (per_level_heads, per_level_anchors) in enumerate(zip(obj_reg_cls_heads, batch_anchors)):
            B, H, W, _, _ = per_level_anchors.shape
            for _ in range(self.per_level_num_anchors):
                feature_hw.append([H, W])

            # TODO: 这里需要理解一下为什么
            if layer_idx == 0:
                for _ in range(self.per_level_num_anchors):
                    per_layer_prefix_ids.append(H * W * self.per_level_num_anchors)
                    previous_layer_prefix = H * W * self.per_level_num_anchors
            elif layer_idx < len(batch_anchors) - 1:
                for _ in range(self.per_level_num_anchors):
                    cur_layer_prefix = H * W * self.per_level_num_anchors
                    per_layer_prefix_ids.append(previous_layer_prefix + cur_layer_prefix)
                previous_layer_prefix = previous_layer_prefix + cur_layer_prefix

            # obj target init value=0
            per_level_obj_target = torch.zeros((B, H * W * self.per_level_num_anchors, 1),
                                               dtype=torch.float32,
                                               device=device)
            # noobj target init value=0
            per_level_noobj_target = torch.ones((B, H * W * self.per_level_num_anchors, 1),
                                                dtype=torch.float32,
                                                device=device)
            # box loss scale init value=0
            per_level_box_loss_scale = torch.zeros((B, H * W * self.per_level_num_anchors, 1),
                                                   dtype=torch.float32,
                                                   device=device)
            # reg target init value=0
            per_level_reg_target = torch.zeros((B, H * W * self.per_level_num_anchors, 4),
                                               dtype=torch.float32,
                                               device=device)
            # cls target init value=-1
            per_level_cls_target = torch.ones((B, H * W * self.per_level_num_anchors, 1),
                                              dtype=torch.float32,
                                              device=device) * (-1)
            # per_level_targets shape is [B, H*W*self.per_level_num_anchors, 8]
            # 6: [obj_target, x_offset, y_offset, scaled_gt_w, scaled_gt_h, class_target]
            per_level_targets = torch.cat((
                per_level_obj_target, per_level_noobj_target, per_level_box_loss_scale,
                per_level_reg_target, per_level_cls_target), dim=-1)

            # per level anchor shape: [B, H*W*3, 5]
            # 5: [grids_x_idx, grids_y_idx, relative_anchor_w, relative_anchor_h, stride]
            per_level_anchors = per_level_anchors.view(
                per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])

            # per_level_heads: [B, H*W*3, 85]
            per_level_heads = per_level_heads.view(
                per_level_anchors.shape[0], -1, per_level_anchors.shape[-1])
            per_level_obj_preds = per_level_heads[..., 0]
            per_level_cls_preds = per_level_heads[..., 5:]

            # per_level_heads[..., 1:3] 是相对于左上角的偏移量
            # per_level_anchors这里anchors里的数值已经相对feature map做了缩小,在anchor.py中
            per_level_scales_xy_ctr = per_level_heads[..., 1:3] + per_level_anchors[..., :2]
            per_level_scales_wh = torch.exp(per_level_heads[..., 3:5]) * per_level_heads[..., 3:5]

            # traverse each pic in batch
            for idx, one_image_annots in enumerate(annotations):
                # drop all index=-1 in annotations  [N, 5]
                one_image_annots = one_image_annots[one_image_annots[:, 4] >= 0]
                # not empty
                if one_image_annots.shape[0] > 0:
                    gt_boxes = one_image_annots[:, :4]
                    gt_classes = one_image_annots[:, 4]

                    scaled_gt_boxes = gt_boxes / all_strides[idx]
                    scaled_gt_wh = scaled_gt_boxes[:, 2:] - scaled_gt_boxes[:, :2]
                    scaled_gt_ctr = scaled_gt_boxes[:, :2] + scaled_gt_wh
                    layer_anchor_sizes = anchor_sizes[idx]

        return all_anchors, all_targets
