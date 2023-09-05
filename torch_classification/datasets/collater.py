import numpy as np
import torch


class Collater:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, datas):
        images = []
        labels = []
        for d in datas:
            images.append(d["image"])
            labels.append(d["label"])

        # [B, H, W, 3]
        images = np.stack(images, axis=0)
        mean = np.asarray(self.mean, dtype=np.float32).reshape((1, 1, 1, 3))
        std = np.asarray(self.std, dtype=np.float32).reshape((1, 1, 1, 3))
        images = ((images / 255.) - mean) / std
        images = torch.from_numpy(images).float()
        images = images.permute(0, 3, 1, 2).contiguous()

        labels = torch.from_numpy(np.array(labels)).long()

        return {
            'image': images,
            'label': labels
        }
