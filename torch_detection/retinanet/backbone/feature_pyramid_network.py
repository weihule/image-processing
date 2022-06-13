from abc import ABC

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
        if set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()

        # 只保存layer4及其之前的结构, 弃掉之后的结构
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        self.return_layers = orig_return_layers





