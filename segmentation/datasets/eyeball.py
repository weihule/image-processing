import os
from pathlib import Path
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


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
        manual = np.array(manual) / 255

        roi_mask = Image.open(self.roi_mask[item]).convert('L')
        roi_mask = 255 - np.array(roi_mask)

        # 眼球部分为0, 血丝部分为1，背景为255
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 传为PIL格式
        mask = Image.fromarray(mask)

        if self.transform:
            img, mask = self.transform(img, mask)


    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    ro = r"D:\workspace\data\dl\DRIVE"
    eye = EyeballDataset(root=ro)
    eye[10]



