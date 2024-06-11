from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset


labels = ['below', 'NoChamfer', 'rough', 'normal']


class MultiDataset(Dataset):
    def __init__(self, root, set_name='train', transform=None):
        self.root = root
        self.transform = transform
        self.set_name = set_name
        self.image_labels = self.total_images()

    def __getitem__(self, item):
        image_path = self.image_labels[item][0]
        image_label = self.image_labels[item][1]
        image = self.get_image(image_path)
        label = image_label

        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_folder2label(self):
        """
        获取每个文件夹对应的多标签
        """
        dir_path = Path(self.root) / self.set_name
        label_dict = {}
        for folder in dir_path.iterdir():
            folder_name = folder.parts[-1]
            mask_label = [0] * len(labels)
            labels_index = [labels.index(i) for i in folder_name.split('-')]
            for i in labels_index:
                mask_label[i] = 1
            label_dict[folder_name] = mask_label
        return label_dict

    @staticmethod
    def get_image(image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return image

    def total_images(self):
        label_dict = self.get_folder2label()
        image_root = Path(self.root) / self.set_name
        image_labels = []
        for folder in image_root.iterdir():
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            folder_name = folder.parts[-1]
            for i in images:
                image_labels.append([str(i), label_dict[folder_name]])
        return image_labels

    def __len__(self):
        return len(self.image_labels)


def main():
    md = MultiDataset(root=r"D:\workspace\data\mojiao")
    content = md[10]
    image = content["image"]
    label = content["label"]
    print(image.shape, label)


if __name__ == "__main__":
    main()


