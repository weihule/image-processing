import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from .resnet50_fpn_model import resnet50_fpn_backbone
from .anchors import RetinaAnchors


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: Tensor):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))

        x = self.output(x)      # [B, num_anchors*4, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, num_anchors*4, H, W] -> [B, H, W, num_anchors*4]
        x = x.view(x.shape[0], -1, 4)   # [B, H*W*num_anchors, 4]

        return x


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()      # 1/(np.exp(-x)+1)

    def forward(self, x: Tensor):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x)) 

        out1 = self.output(x)      # [B, num_anchors*num_classes, H, W]
        out1 = self.output_act(out1)  # [B, num_anchors*num_classes, H, W]
        b, c, h, w = out1.shape

        out2 = out1.view(b, h, w, self.num_anchors, self.num_classes).contiguous()
        out2 = out2.view(b, -1, self.num_classes)     # [B, num_anchors*H*W, num_classes]

        return out2


class RetinaNet(nn.Module):
    def __init__(self, num_classes):
        super(RetinaNet, self).__init__()
        self.fpn_backbone = resnet50_fpn_backbone()
        self.regressionModel = RegressionModel(num_features_in=256)

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.anchors = RetinaAnchors()

    def forward(self, inputs):
        batch_size = inputs.shape[0]
        features = self.fpn_backbone(inputs)

        fpn_feature_sizes = list()
        cls_heads, reg_heads = list(), list()
        for feature in features:
            fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])

            # cls_head shape: [B, 9*H*W, num_classes]
            cls_head = self.classificationModel(feature)
            cls_heads.append(cls_head)

            # reg_head shape: [B, 9*H*W, 4]
            reg_head = self.regressionModel(feature)
            reg_heads.append(reg_head)

        del features

        fpn_feature_sizes = torch.tensor(fpn_feature_sizes)
        # if input size: [B, 3, 640, 640]
        # batch_anchors shape: [[B, 80*80*9, 4], [B, 40*40*9, 4], [B, 20*20*9, 4], [B, 10*10*9, 4], [B, 5*5*9, 4]]
        batch_anchors = self.anchors(batch_size, fpn_feature_sizes)

        return cls_heads, reg_heads, batch_anchors


if __name__ == "__main__":
    inputs = torch.rand(4, 3, 640, 640)
    # backbone = resnet50_fpn_backbone()
    # res = backbone(inputs)
    # for p in res:
    #     print(p.shape)

    model = RetinaNet(num_classes=80)
    pred_cls_heads, pred_reg_heads, pred_batch_anchors = model(inputs)
    for pred_cls_head, pred_reg_head, pred_batch_anchor in zip(
            pred_cls_heads, pred_reg_heads, pred_batch_anchors
    ):
        print(pred_cls_head.shape, pred_reg_head.shape, pred_batch_anchor.shape)



