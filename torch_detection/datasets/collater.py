import numpy as np
import torch

__all__ = [
    "DetectionCollater"
]


class DetectionCollater:
    def __init__(self, divisor=32):
        self.divisor = divisor

    def __call__(self, datas):
        images = [s['image'] for s in datas]
        annots = [s['annots'] for s in datas]
        scales = [s['scale'] for s in datas]
        sizes = [s['size'] for s in datas]

        max_h = max(image.shape[0] for image in images)
        max_w = max(image.shape[1] for image in images)

        pad_h = 0 if max_h % self.divisor == 0 else self.divisor - max_h % self.divisor
        pad_w = 0 if max_w % self.divisor == 0 else self.divisor - max_w % self.divisor
        input_images = np.zeros((len(images), max_h + pad_h, max_w + pad_w, 3),
                                dtype=np.float32)
        input_images = torch.from_numpy(input_images)
        # [B,H,W,3] -> [B,3,H,W]
        input_images = input_images.permute(0, 3, 1, 2).contiguous()

        max_annots_num = max(annot.shape[0] for annot in annots)
        if max_annots_num > 0:
            input_annots = np.ones(
                (len(annots), max_annots_num, 5), dtype=np.float32
            )*(-1)
            for i, annot in annots:
                if annot.shape[0] > 0:
                    input_annots[i, :annot.shape[0], :] = annot
        else:
            input_annots = np.ones(
                (len(annots), 1, 5), dtype=np.float32) * (-1)
        input_annots = torch.from_numpy(input_annots)

        scales = np.array(scales, dtype=np.float32)
        sizes = np.array(sizes, dtype=np.float32)

        return {
            'image': input_images,
            'annots': input_annots,
            'scale': scales,
            'size': sizes,
        }
