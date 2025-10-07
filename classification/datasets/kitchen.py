from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset

__all__ = [
    "labels_list",
    "KitchenDataset"
]

# labels_list = ['normal', 'smoke', 'shirtless', 'rat', 'cat', 'dog']
labels_list = ['normal', 'uneven', 'unfinished_chamfers', 'below']


class KitchenDataset(Dataset):
    def __init__(self,
                 root_dir,
                 set_name='train',
                 transform=None):
        super(KitchenDataset, self).__init__()
        assert set_name in ["train", "val"], "wrong set_name !"
        if not Path(root_dir).exists():
            raise FileExistsError(f"{root_dir} not exists !")

        self.image_dir = Path(root_dir) / set_name

        img_paths, labels = self.pre_process()
        self.img_paths, self.labels = img_paths, labels

        self.transform = transform
        print("=> {}".format(set_name))
        print("| Dataset Size      |{:<8d} |".format(len(self.img_paths)))
        print("| Dataset Class Num |{:<8d} |".format(len(labels_list)))

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

    def pre_process(self):
        images = []
        labels = []
        # img_paths = [x for x in Path(self.image_dir).glob("*.jpg")] + [x for x in Path(self.image_dir).glob("*.png")]
        for per_dir in self.image_dir.iterdir():
            label = labels_list.index(per_dir.parts[-1])
            img_paths = [x for x in per_dir.glob("*.jpg")] + [x for x in
                                                              per_dir.glob("*.png")]
            for per_img in per_dir.iterdir():
                images.append(str(per_img))
                labels.append(label)
        return images, labels

    def load_image(self, item):
        image_path = self.img_paths[item]
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, item):
        label = self.labels[item]

        return np.asarray(label, dtype=np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kit = KitchenDataset(root_dir=r"D:\workspace\data\kitchen",
                         set_name="train")
    s = kit[0]
    img, lab = s["image"], s["label"]
    print(lab, labels_list[int(lab)])
    plt.imshow(img.astype(np.uint8))
    plt.show()
