from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)


class BaseTransform:
    def __init__(self) -> None:
        pass

    def apply_image(self, labels):
        pass

    def apply_instances(self, labels):
        pass

    def apply_semantic(self, labels):
        pass

    def __call__(self, labels):
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)
    

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    
    def append(self, transform):
        self.transforms.append(transform)
    
    def insert(self, index, transform):
        self.transforms.insert(index, transform)

    def __getitem__(self, index: list | int) -> Compose:
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        return Compose([self.transforms[i] for i in index]) if isinstance(index, list) else self.transforms[index]
    
    def __setitem__(self, index: list|int, value: list|int) -> None:
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(value, list), (
                f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
            )
        if isinstance(index, int):
            index, calue = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        return self.transforms
    
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform:
    """
    Examples:
        >>> dataset = YOLODataset("path/to/data")
        >>> pre_transform = Compose([RandomFlip(), RandomPerspective()])
        >>> mix_transform = BaseMixTransform(dataset, pre_transform, p=0.5)
    """
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p
    
    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        """
        Examples:
            >>> transform = BaseMixTransform(dataset, pre_transform=None, p=0.5)
            >>> result = transform({"image": img, "bboxes": boxes, "cls": classes})
        """
        if random.uniform(0, 1) > self.p:
            return labels
        
        # Get index of onr or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic, CutMix or Mixup
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        labels = self._update_label_text(labels)

    def get_indexes(self):
        return random.randint(0, len(self.dataset)-1)
    
    @staticmethod
    def _update_label_text(labels: dict[str, Any]) -> dict[str, Any]:
        if "texts" not in labels:
            return labels
        mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels
