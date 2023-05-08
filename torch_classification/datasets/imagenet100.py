import os
import json

import cv2
import numpy as np
from torch.utils.data import Dataset
from transform import transform


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

        print(f"Dataset size: {len(self.image_path_list)}")
        print(f"Dataset Class Num: {len(self.class2label)}")

    def __getitem__(self, index):
        image = self.load_image(index)
        label = self.load_label(index)

        cv2.namedWindow("inside", cv2.WINDOW_FREERATIO)
        cv2.imshow("inside", cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
        cv2.waitKey(0)

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


if __name__ == "__main__":
    root_dir_ = r"D:\workspace\data\dl\imagenet100"
    imagenet = ImageNet100Dataset(root_dir=root_dir_,
                                  transform=transform["train"])
    sample_ = imagenet[90]
    
    image = cv2.cvtColor(sample_["image"], cv2.COLOR_RGB2BGR).astype(np.uint8)
    cv2.namedWindow("res", cv2.WINDOW_FREERATIO)
    cv2.imshow("res", sample_["image"])
    cv2.waitKey(0)
    print(sample_["label"])



