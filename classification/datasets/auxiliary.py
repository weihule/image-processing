import os
import json
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset

AuxiliaryClass = [
    "人员照片",
    "偏航减速机",
    "偏航制动盘",
    "冷却管路",
    "动力电缆",
    "发电机散热器上",
    "发电机散热器下",
    "发电机集油盒",
    "变桨轴承",
    "变流器检查",
    "塔基水冷系统与发电机水冷压力表",
    "塔筒螺栓",
    "润滑油量",
    "滑环",
    "联轴器",
    "齿轮箱润滑管路",
]


class AuxiliaryDataset(Dataset):
    def __init__(self,
                 root_dir,
                 set_name='train',
                 transform=None):
        super(AuxiliaryDataset, self).__init__()
        assert set_name in ["train", "val"], "wrong set_name !"
        if not Path(root_dir).exists():
            raise FileExistsError(f"{root_dir} not exists !")

        img_paths, labels = self.prepare_data(root_dir, set_name)
        self.img_paths, self.labels = img_paths, labels

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
    def prepare_data(root_dir, set_name):
        # 获取图片路径 和 对应的label
        paths, labels = [], []
        for idx, sub_class_dir in enumerate(Path(root_dir).joinpath(set_name).iterdir()):
            label = AuxiliaryClass.index(sub_class_dir.parts[-1])
            for per_image_path in sub_class_dir.iterdir():
                if per_image_path.exists():
                    paths.append(str(per_image_path))
                    labels.append(label)
                else:
                    continue
        return paths, labels

    def load_image(self, item):
        image_path = self.img_paths[item]
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, item):
        label = self.labels[item]

        return np.asarray(label, dtype=np.float32)


def test():
    image_root = r"D:\workspace\MyData\巡检230104test1"
    set_name = "train"
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    aux = AuxiliaryDataset(root_dir=image_root,
                           set_name=set_name,
                           transform=None)
    sample = aux[10]
    image, label = sample["image"], sample["label"]
    print(label, AuxiliaryClass[int(label)], image.shape)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("res", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    test()
