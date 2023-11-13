import os
from pathlib import Path

import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def collate_fn(batch):
    """
    在collate_fn中将PIL转为numpy, 再转为Tensor
    Args:
        batch:

    Returns:

    """
    max_size_w = max(pi["image"].size(0) for pi in batch)
    max_size_h = max(pi["image"].size(1) for pi in batch)

    images = []
    labels = []
    for d in batch:
        # 创建一个新的空白图像，用0值填充
        new_img = Image.new('RGB', (max_size_w, max_size_h),
                            color=(0, 0, 0))

        # 将原始图像粘贴到新图像上
        new_img.paste(d["image"], (0, 0))
        images.append(np.array(new_img))

        new_mask = Image.new('L', (max_size_w, max_size_h),
                             color=(255, 255, 255))
        new_mask.paste(d["mask"], (0, 0))
        labels.append(np.array(new_mask))

    batch_images = np.stack(images, axis=0).transpose((0, 3, 1, 2))
    batch_masks = np.stack(labels, axis=0)
    return {"image": batch_images, "mask": batch_masks}


class EyeballDataset(Dataset):
    def __init__(self, root, set_name="training", transform=None):
        super(EyeballDataset, self).__init__()
        assert set_name in ["training", "test"], "Error set_name"
        data_root = Path(root) / set_name
        assert data_root.exists(), f"path {str(data_root)} does not exists"
        self.transform = transform
        self.img_list = list(data_root.joinpath("images").glob("*.tif"))
        self.manual = list(data_root.joinpath("1st_manual").glob("*.gif"))
        self.roi_mask = list(data_root.joinpath("mask").glob("*.gif"))

    def __getitem__(self, item):
        image = Image.open(self.img_list[item]).convert("RGB")
        manual = Image.open(self.manual[item]).convert("L")

        # 把血丝部分(白色255)的像素变为1, 其余部分是黑色, 所以还是0
        manual = np.array(manual) / 255

        roi_mask = Image.open(self.roi_mask[item]).convert('L')
        # 把眼球(白色)变为0, 其余部分为255 - 0 = 255
        roi_mask = 255 - np.array(roi_mask)

        # 眼球部分为0, 血丝部分为1，背景为255(白)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 传为PIL格式
        mask = Image.fromarray(mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        return {"image": image, "mask": mask}

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    ro = r"D:\workspace\data\dl\eyeball"
    eye = EyeballDataset(root=ro)
    ds = eye[10]
    i, m = ds["image"], ds["mask"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    # 在第一个子图上绘制 data1
    axes[0].imshow(i)
    axes[0].axis('off')  # 隐藏坐标轴信息

    axes[1].imshow(i.transpose(Image.FLIP_LEFT_RIGHT))
    axes[1].axis('off')

    axes[2].imshow(i.transpose(Image.FLIP_TOP_BOTTOM))
    axes[2].axis('off')

    plt.show()
