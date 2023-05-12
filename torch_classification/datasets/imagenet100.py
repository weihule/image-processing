import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = [
    "ImageNet100Dataset"
]


class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir, set_name='imagenet100_train', transform=None):
        super(ImageNet100Dataset, self).__init__()
        assert set_name in ['imagenet100_train', 'imagenet100_val'], 'Wrong set_name'
        set_dir = os.path.join(root_dir, set_name)

        sub_class_name_list = list()
        for sub_class_name in os.listdir(set_dir):
            sub_class_name_list.append(sub_class_name)
        sub_class_name_list = sorted(sub_class_name_list)

        self.image_path_list = list()
        for sub_class_name in os.listdir(set_dir):
            per_class_dir = os.path.join(set_dir, sub_class_name)
            for per_image_name in os.listdir(per_class_dir):
                per_image_path = os.path.join(per_class_dir, per_image_name)
                self.image_path_list.append(per_image_path)

        self.class2label = {
            sub_class: i
            for i, sub_class in enumerate(sub_class_name_list)
        }

        self.label2class = {
            v: k
            for k, v in self.class2label.items()
        }

        self.transform = transform
        print("=> {}".format(set_name))
        print("| Dataset Size      |{:<8d} |".format(len(self.image_path_list)))
        print("| Dataset Class Num |{:<8d} |".format(len(self.class2label)))

    def __getitem__(self, index):
        image = self.load_image(index)
        label = self.load_label(index)

        # cv2.namedWindow("inside", cv2.WINDOW_FREERATIO)
        # cv2.imshow("inside", cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # cv2.waitKey(0)

        sample = {
            "image": image,
            "label": label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_path_list)

    def load_image(self, idx):
        image_path = self.image_path_list[idx]
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, idx):
        image_name = self.image_path_list[idx].split(os.sep)[-2]
        label = self.class2label[image_name]

        return np.asarray(label, dtype=np.float32)


def test(imagenet):
    sample_ = imagenet[90]

    image_ = sample_["image"]
    print(type(image_), image_.shape)
    image_ = cv2.cvtColor(image_, cv2.COLOR_RGB2BGR).astype(np.uint8)
    cv2.namedWindow("res", cv2.WINDOW_FREERATIO)
    cv2.imshow("res", image_)
    cv2.waitKey(0)
    print(sample_["label"])


def test02(dataset):
    collater = Collater(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5))
    loader = DataLoader(dataset=dataset,
                        batch_size=16,
                        shuffle=True,
                        collate_fn=collater)
    print("len(loader) = ", len(loader))
    for datas in loader:
        images = datas["image"]
        labels = datas["label"]
        print(images.shape, labels)


if __name__ == "__main__":
    from transform import transform, Collater
    root_dir_ = r"D:\workspace\data\dl\imagenet100"
    imagenet_ = ImageNet100Dataset(root_dir=root_dir_,
                                   set_name='imagenet100_val',
                                   transform=transform["train"])
    test02(imagenet_)

