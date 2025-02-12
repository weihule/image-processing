import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

from backbone.resnet_backbone import resnet50


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        # print(f"... inputs1.shape = {inputs1.shape} and inputs2.shape = {inputs2.shape}")
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        # print(f"------ outputs.shape = {outputs.shape}")
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class UNet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone = 'resnet50'):
        super(UNet, self).__init__()
        self.backbone = backbone
        if backbone == "resnet50":
            self.resnet = resnet50()
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        # upsampling
        self.up_concat4 = UNetUp(in_filters[3], out_filters[3])
        self.up_concat3 = UNetUp(in_filters[2], out_filters[2])
        self.up_concat2 = UNetUp(in_filters[1], out_filters[1])
        self.up_concat1 = UNetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

    def forward(self, inputs):
        # 以 inputs 为 [B, 3, 640, 640] 为例
        if self.backbone == "resnet50":
            # [4, 64, 320, 320], [4, 256, 160, 160], [4, 512, 80, 80], [4, 1024, 40, 40], [4, 2048, 20, 20]
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

        # print(f"feat1 shape: {feat1.shape}")
        # print(f"feat2 shape: {feat2.shape}")
        # print(f"feat3 shape: {feat3.shape}")
        # print(f"feat4 shape: {feat4.shape}")
        # print(f"feat5 shape: {feat5.shape}")

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        # print(f"up4 shape: {up4.shape}")
        # print(f"up3 shape: {up3.shape}")
        # print(f"up2 shape: {up2.shape}")
        # print(f"up1 shape: {up1.shape}")
        if self.up_conv != None:
            up1 = self.up_conv(up1)

        final = self.final(up1)

        return final

    def freeze_backbone(self):
        if self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "resnet50":
            for param in self.resnet.parameters():
                param.requires_grad = True


def test():
    unet = UNet()
    inputs = torch.randn(4, 3, 640, 640)

    # [batch_size, num_classes, h, w]
    outs = unet(inputs)
    print(outs.shape)

if __name__ == "__main__":
    test()
