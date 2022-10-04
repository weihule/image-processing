import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .other_modules import HorizontalPooling


__all__ = [
    'resnet50',
]


class ResNet50(nn.Module):
    def __init__(self, num_classes, loss=None, aligned=False, **kwargs):
        super(ResNet50, self).__init__()
        if loss is None:
            self.loss = {'softmax_cent'}
        else:
            self.loss = loss

        resnet50 = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feature_dim = 2048     # feature dimension
        self.aligned = aligned
        self.horizontal_pool = HorizontalPooling()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)    # [batch_size, 2048, 8, 4]
        if not self.training:
            lf = self.horizontal_pool(x)    # [batch_size, 2048, 8, 1]
        # 训练阶段，使用aligned
        if self.aligned and self.training:
            lf = x.clone()
            lf = self.bn(lf)    # [batch_size, 2048, 8, 4]
            lf = self.relu(lf)
            lf = self.horizontal_pool(lf)    # [batch_size, 2048, 8, 1]
            lf = self.conv1(lf)  # [batch_size, 128, 8, 1]

        # 如果使用的aligned，或者处于测试阶段的时候
        if self.aligned or not self.training:
            lf = lf.view(lf.shape[:3])  # [batch_size, 128, 8]
            # TODO：没明白为什么会有这么一步操作
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.shape[2:])    # [batch_size, 2048, 1, 1]
        f = x.view(x.shape[0], -1)    # [batch_size, 2048]

        # 测试阶段
        if not self.training:
            return f, lf
        y = self.classifier(f)

        # y: [batch_size, num_classes]
        # f: [batch_size, 2048]
        # lf: [batch_size, 128, 8]
        if self.loss == 'softmax_cent':
            return y, f
        elif self.loss == 'softmax_trip' and not self.aligned:
            return y, f
        elif self.loss == 'softmax_trip' and self.aligned:
            return y, f, lf
        else:
            raise KeyError(f'unknown loss type {self.loss}')


"""ResNet"""


def resnet50(num_classes, loss='softmax_cent', aligned=False, **kwargs):
    model = ResNet50(
        num_classes=num_classes,
        loss=loss,
        aligned=aligned,
        **kwargs
    )
    return model


if __name__ == "__main__":
    import torch
    arr = torch.randn(4, 128, 8)
    print(arr.sum(dim=1, keepdim=True).shape)


