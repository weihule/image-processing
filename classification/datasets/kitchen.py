from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset

__all__ = [
    "labels_list",
    "KitchenDataset"
]


labels_list = ['normal', 'smoke', 'shirtless', 'rat', 'cat', 'dog']


class KitchenDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.images, self.labels = self.pre_process()
        if transform is not None:
            self.transform = transform

    def __getitem__(self, item):
        image = self.load_image(item)
        label = self.load_label(item)

        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, item):
        image_path = self.images[item]
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, item):
        label = self.labels[item]

        return np.asarray(label, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def pre_process(self):
        images = []
        labels = []
        for per_dir in Path(self.root).iterdir():
            label = labels_list.index(per_dir.parts[-1])
            for per_img in per_dir.iterdir():
                images.append(str(per_img))
                labels.append(label)
        return images, labels


if __name__ == "__main__":
    kit = KitchenDataset(root="/home/8TDISK/weihule/data/kitchen")


