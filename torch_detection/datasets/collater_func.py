import os
import torch
import cv2
import random
from .data_transfrom import Normalize

__all__ = [
    'collater',
    'MultiScaleCollater'
]


def collater(datas):
    normalizer = Normalize()
    batch_size = len(datas)
    imgs = [p['img'] for p in datas]
    # annots 里面每一个元素都是一个二维数组, [n, 5]
    annots = [torch.from_numpy(p['annot']) for p in datas]
    scales = [p['scale'] for p in datas]

    img_h_list = [p.shape[0] for p in imgs]
    img_w_list = [p.shape[1] for p in imgs]
    max_h, max_w = max(img_h_list), max(img_w_list)
    padded_img = torch.full((batch_size, max_h, max_w, 3), 111, dtype=torch.float32)
    for i in range(batch_size):
        img = imgs[i]
        padded_img[i, :img.shape[0], :img.shape[1], :] = torch.from_numpy(img)

    # 归一化，标准化
    padded_img = padded_img / 255.
    padded_img = normalizer(padded_img)
    padded_img = padded_img.permute(0, 3, 1, 2).contiguous()

    max_num_annots = max([annot.shape[0] for annot in annots])
    if max_num_annots > 0:
        padded_annot = torch.ones((len(datas), max_num_annots, 5)) * (-1)
        for idx, anno in enumerate(annots):
            if anno.shape[0] > 0:
                padded_annot[idx, :anno.shape[0], :] = anno
    else:
        padded_annot = torch.ones((len(datas), 1, 5)) * (-1)

    return {'img': padded_img, 'annot': padded_annot, 'scale': scales}


"""
这里采用yolov3/yolov5中的multi scale方法
首先确定一个最小到最大尺寸的范围,比如没用multi scale前resize=416,
那么可以确定一个范围[0.5,1.5],乘以416之后就是最小到最大尺寸的范围。
然后,选择一个stride长度,比如yolov3/yolov5是32,
找出最小到最大尺寸中所有能被stride长度整除的尺寸。
最后,随机选择一个尺寸,将所有图片resize到该尺寸再填充成正方形图片就可以送入网络训练了。
注意用multi scale时transform里不再使用resize方法
"""


class MultiScaleCollater():
    def __init__(self,
                 mean,
                 std,
                 resize=416,
                 multi_scale_range=None,
                 stride=32,
                 use_multi_scale=False,
                 normalize=False):
        self.mean = torch.tensor(mean, dtype=torch.float32).tile(1, 1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).tile(1, 1, 1, 1)
        self.resize = resize
        if multi_scale_range is None:
            self.multi_scale_range = [0.5, 1.5]
        else:
            self.multi_scale_range = multi_scale_range
        self.stride = stride
        self.use_multi_scale = use_multi_scale
        self.normalize = normalize

    def __call__(self, samples):
        if self.use_multi_scale:
            min_resize = int(
                ((self.resize + self.stride) * self.multi_scale_range[0]) //
                self.stride * self.stride)
            max_resize = int(
                ((self.resize + self.stride) * self.multi_scale_range[1]) //
                self.stride * self.stride)
            final_resize = random.choice(range(min_resize, max_resize, self.stride))
        else:
            final_resize = self.resize

        imgs = [p['img'] for p in samples]
        annots = [p['annot'] for p in samples]
        scales = [p['scale'] for p in samples]

        batch_size = len(imgs)

        padded_img = torch.full((batch_size, final_resize, final_resize, 3), 111)

        for index, img in enumerate(imgs):
            height, width, _ = img.shape
            resize_factor = final_resize / max(height, width)
            origin_h, origin_w = int(resize_factor * height), int(resize_factor * width)
            img = cv2.resize(img, (origin_w, origin_h))
            padded_img[index, :origin_h, :origin_w, :] = torch.from_numpy(img)

            annots[index][:, :4] *= resize_factor
            scales[index] *= resize_factor

        # padded_img [B, H, W, 3] -> [B, 3, H, W]
        padded_img = padded_img / 255.
        if self.normalize:
            padded_img = (padded_img - self.mean) / self.std
        padded_img = padded_img.permute(0, 3, 1, 2).contiguous()

        max_num_annots = max(annot.shape[0] for annot in annots)
        if max_num_annots > 0:
            padded_annots = torch.ones((batch_size, max_num_annots, 5)) * (-1)

            for annot_index, p in enumerate(annots):
                if p.shape[0] > 0:
                    padded_annots[annot_index, :p.shape[0], :] = torch.from_numpy(p)
        else:
            padded_annots = torch.ones((batch_size, 1, 5)) * (-1)

        return {'img': padded_img, 'annot': padded_annots, 'scale': scales}



