import os
import json
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset


class Flower5(Dataset):
    def __init__(self,
                 root_dir,
                 set_name='train',
                 class_file=None,
                 transform=None):
        super(Flower5, self).__init__()
        assert set_name in ["train", "val"], "wrong set_name !"
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


def test02():
    root_dir = r"D:\workspace\data\dl\flower"
    set_name = "train"
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    flower5 = Flower5(root_dir=root_dir,
                      set_name=set_name,
                      transform=None,
                      class_file=class_file)
    sample = flower5[0]
    image, label = sample["image"], sample["label"]
    print(label, flower5.idx2cls[int(label)], image.shape)
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("res", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    test02()
