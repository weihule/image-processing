import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from typing import List
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, img_path: List, label_path: List, transform=None):
        super(CustomDataset, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item])
        if img.mode != "RGB":
            raise ValueError("image {} isn't RGB mode".format(self.img_path[item]))
        label = self.label_path[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)

        return images, labels


