import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
import torch.nn.functional as F
from typing import Tuple, List, Dict


class IntermediateLayerGetter(nn.ModuleDict):
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        super(IntermediateLayerGetter, self).__init__()
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 遍历模型子模块按顺序存入有序字典
        # 只保存layer4及其之前的结构, 弃掉之后的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            # 如果return_layers为空,停止
            if len(return_layers) == 0:
                break

        # {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        self.return_layers = orig_return_layers
        self.layers = layers

    def forward(self, x):
        out = OrderedDict()
        # 依次遍历模型的所有子模块，并进行正向传播，
        # 收集layer1, layer2, layer3, layer4的输出
        for name, module in self.layers.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class BackboneWithFPN(nn.Module):
    """
    Adds a FPN on top of a model.
    Internally, it uses torchvision.models._utils.IntermediateLayerGetter to
    extract a submodel that returns the feature maps specified in return_layers.
    The same limitations of IntermediatLayerGetter apply here.
    """

    def __init__(self,
                 backbone: nn.Module,
                 return_layers=None,
                 in_channels_list=None,     # resnet50提供给fpn的特征层channels [256, 512, 1024, 2048]
                 out_channels=256,
                 extra_blocks=None,
                 re_getter=True):
        super(BackboneWithFPN, self).__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if re_getter:
            assert return_layers is not None
            # 得到一个 OrderedDict(), key 是 0,1,2,3; value对应的是 layer1-layer4的输出
            self.body = IntermediateLayerGetter(backbone, return_layers)    
        else:
            self.body = backbone

        self.fpn = FeaturePyramidNetwork(in_channels_list,
                                         out_channels,
                                         extra_blocks)


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.
    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
             original feature maps

    Returns:
        results (List[Tensor]):the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def forward(self,
                results: List[Tensor],
                x: List[Tensor],
                names: List[str]) -> Tuple[List[Tensor], List[str]]:
        pass


class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        # x.append(F.max_pool2d(x[-1], 1, 2, 0))
        x.append(F.max_pool2d(x[-1], kernel_size=1, stride=2, padding=0))
        return x, names


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(LastLevelP6P7, self).__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=2, padding=1)
        self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=2, padding=1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self,
                p: List[Tensor],
                c: List[Tensor],
                names: List[str]) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])

        return p, names


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        # in_channels_list 就是 resnet50 的输出层channel [256, 512, 1024, 2048]
        super(FeaturePyramidNetwork, self).__init__()

        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel (kernel_size=1)
        self.inner_blocks = nn.ModuleList()

        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()

        # in_channels 分别是256, 512, 1024, 2048
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, kernel_size=1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

            # initialize parameters now to avoid modifying the initialization of top_blocks
            for m in self.children():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, a=1)
                    nn.init.constant_(m.bias, 0)

            self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        # 这里idx取-1, 即直接取 self.inner_blocks
        # 最后一个模块
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        :param x: (OrderedDict[Tensor]): feature maps for each feature level.
        :return:
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        # 将resnet layer4的channel调整到指定的out_channels
        # 如果输入是 224*224*3, 且 in_channels_list 是
        # [256, 512, 1024, 2048], 那么调整之后 last_inner 就是
        # 7*7*256
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)

        # result中保存着每个预测特征层
        results = list()

        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        # 如果 len(x) = 4, 则idx是 2, 1, 0
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            # inner_lateral.shape 是 [B,C,H,W]
            # [-2:] 就是得到了 H, W
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, feat_shape, mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out



