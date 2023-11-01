import os
import json
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

__all__ = [
    "ImageNet100Dataset",
    "ImageNet100"
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


class ImageNet100(Dataset):
    def __init__(self,
                 root_dir,
                 set_name='imagenet100_train',
                 class_file=None,
                 transform=None):
        super(ImageNet100, self).__init__()
        assert set_name in ["imagenet100_train", "imagenet100_val"], "wrong set_name !"
        if not Path(root_dir).exists():
            raise FileExistsError(f"{root_dir} not exists !")

        if class_file is None or not Path(class_file).exists():
            raise FileNotFoundError(f"{class_file} not found !")

        img_paths, labels, idx2cls = self.prepare_data(root_dir, set_name, class_file)
        self.img_paths, self.labels, self.idx2cls = img_paths, labels, idx2cls

        self.transform = transform
        print("=> {}".format(set_name))
        print("| Dataset Size      |{:<8d} |".format(len(self.img_paths)))
        print("| Dataset Class Num |{:<8d} |".format(len(self.labels)))

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

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def prepare_data(root_dir, set_name, class_file):
        # 获取各类别的索引
        with open(class_file, "r", encoding="utf-8") as fr:
            cls2idx = json.load(fr)
        idx2cls = {v: k for k, v in cls2idx.items()}

        # 获取图片路径 和 对应的label
        paths, labels = [], []
        for idx, sub_class_dir in enumerate(Path(root_dir).joinpath(set_name).iterdir()):
            label = cls2idx[sub_class_dir.parts[-1]]
            labels.extend([label] * len(list(sub_class_dir.glob("*"))))
            for per_image_path in sub_class_dir.iterdir():
                if per_image_path.exists():
                    paths.append(str(per_image_path))
                else:
                    paths.append(-1)
        return paths, labels, idx2cls

    def load_image(self, item):
        image_path = self.img_paths[item]
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, item):
        label = self.labels[item]

        return np.asarray(label, dtype=np.float32)


def main():
    from transform import transform
    root_dir_ = r"D:\workspace\data\dl\imagenet100"
    imagenet_ = ImageNet100Dataset(root_dir=root_dir_,
                                   set_name='imagenet100_val',
                                   transform=transform["train"])


def test02():
    from transform import transform
    root_dir = r"D:\workspace\data\dl\imagenet100"
    set_name = "imagenet100_val"
    class_file = r"D:\workspace\data\dl\imagenet100\class_100.json"
    imgnet100 = ImageNet100(root_dir=root_dir,
                            set_name=set_name,
                            transform=None,
                            class_file=class_file)
    sample = imgnet100[1001]
    image, label = sample["image"], sample["label"]
    print(label, imgnet100.idx2cls[int(label)])
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("res", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # main()
    test02()
