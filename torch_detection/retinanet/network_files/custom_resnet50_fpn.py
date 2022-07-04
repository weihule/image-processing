import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
import torch.nn.functional as F
from typing import Tuple, Callable, List, Dict


class LayerGetter(nn.Module):
    def __init__(self, model, return_layers: Dict):
        super(LayerGetter, self).__init__()
        if not set(return_layers.keys()).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        layers = OrderedDict()
        origin_return_layers = {k: v for k, v in return_layers.items()}
        
        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构, 弃掉之后的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if len(return_layers) == 0:
                break

        self.return_layers = origin_return_layers
        self.layers = layers
        
    def forward(self, x):
        # 遍历self.layers中的模块,
        # 进行正向传播并获得其输出
        out = OrderedDict()
        for name, module in self.layers.items():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x

        return out


class PyramidFeatures(nn.Module):
    def __init__(self, back_bone, return_layers, c3_size, c4_size, c5_size, out_channel):
        super(PyramidFeatures, self).__init__()
        self.back_bone = back_bone
        self.return_layers = return_layers

        # upsample C5 to get P5
        self.p5_1 = nn.Conv2d(c5_size, out_channel, kernel_size=1, stride=1)
        self.p5_up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1)

        # add P5 elementwise to C4
        self.p4_1 = nn.Conv2d(c4_size, out_channel, kernel_size=1, stride=1)
        self.p4_up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1)

        # add P4 elementwise to C3
        self.p3_1 = nn.Conv2d(c3_size, out_channel, kernel_size=1, stride=1)
        self.p3_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.p6 = nn.Conv2d(c5_size, out_channel, kernel_size=3, padding=1, stride=2)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.p7_1 = nn.ReLU()
        self.p7_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        lg = LayerGetter(self.back_bone, {'layer2': '1', 'layer3': '2', 'layer4': '3'})
        out = lg(x)

        input_channels = list()
        input_tensor = list()

        for k, v in out.items():
            input_channels.append(v.shape[1])
            input_tensor.append(v)
        c3, c4, c5 = input_tensor
        # print(c3.shape, c4.shape, c5.shape)

        p5_x = self.p5_1(c5)
        p5_up_sample_x = self.p5_up_sample(p5_x)
        p5_x = self.p5_2(p5_x)

        p4_x = self.p4_1(c4)
        p4_x += p5_up_sample_x
        p4_up_sample_x = self.p4_up_sample(p4_x)
        p4_x = self.p4_2(p4_x)

        p3_x = self.p3_1(c3)
        p3_x += p4_up_sample_x
        p3_x = self.p3_2(p3_x)

        p6_x = self.p6(c5)

        p7_x = self.p7_1(p6_x)
        p7_x = self.p7_2(p7_x)

        return [p3_x, p4_x, p5_x, p6_x, p7_x]

        