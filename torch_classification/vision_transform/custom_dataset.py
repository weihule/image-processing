import typing
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from typing import List
from PIL import Image
import torch
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, img_path: List, label_path: List, transform=None):
        super(CustomDataset, self).__init__()
        self.images_path = img_path
        self.label_path = label_path
        self.transform = transform
        if len(self.images_path) != len(self.label_path):
            raise 'images number not equal lables number'

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        if img.mode != "RGB":
            raise ValueError("image {} isn't RGB mode".format(self.images_path[index]))
        label = self.label_path[index]
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


# if __name__ == "__main__":
    # train_path = 'D:\\workspace\\data\\DL\\flower\\train'
    # train_set = datasets.ImageFolder(train_path)
    # print(len(train_set))
    # print(train_set.class_to_idx)
    # print(train_set.imgs)
    # for i in train_set:
    #     print(i)

    # root = 'D:\\workspace\\data\\DL\\flower'
    # label_file = 'D:\\workspace\\code\\study\\torch_classification\\shuffleNet\\class_indices.txt'
    # train_images_path, train_images_label = read_split_data(root, "train", label_file)
    # CD = Custom_Dataset(train_images_path, train_images_label, None)
    # for i in range(CD.__len__()):
    #     mode = CD.__getitem__(i)
        


