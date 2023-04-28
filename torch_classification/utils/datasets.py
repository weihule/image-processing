import os

import numpy as np
import torch
import torchvision
import cv2
from PIL import Image
from torch.utils.data import Dataset


class FlowerDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        super(FlowerDataset, self).__init__()
        self.images_path = img_path
        self.label_path = label_path
        self.transform = transform
        if len(self.images_path) != len(self.label_path):
            raise 'images number not equal lables number'

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        simple = {}
        img = Image.open(self.images_path[index])
        if img.mode != 'RGB':
            raise ValueError(f'wrong img mode {self.images_path[index]}')
        label = self.label_path[index]
        if self.transform is not None:
            img = self.transform(img)
        simple['img'] = img
        simple['label'] = label

        return simple


def collater(simples):
    images = [p['img'] for p in simples]
    labels = [p['label'] for p in simples]

    images = torch.stack(images, dim=0)
    labels = torch.as_tensor(labels)
    return images, labels


# ======================================
class CustomDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        super(CustomDataset, self).__init__()
        self.images_path = img_path
        self.label_path = label_path
        self.transform = transform
        if len(self.images_path) != len(self.label_path):
            raise 'images number not equal lables number'

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image = self.load_image(idx)
        label = self.load_label(idx)
        sample = {
            'image': image,
            'label': label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, idx):
        image = cv2.imdecode(np.fromfile(self.images_path[idx], dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image.astype(np.float32)

    def load_label(self, idx):
        label = self.label_path[idx]

        return np.float32(label)


class ClassificationCollater:
    def __init__(self,
                 mean,
                 std):
        self.mean = mean
        self.std = std

    def __call__(self, datas):
        images = [np.array(s['image']).astype(np.float32) for s in datas]
        labels = [s['label'] for s in datas]

        # B, H, W, 3
        # stack会增加一个维度
        images = np.stack(images, axis=0)
        mean = np.expand_dims(np.expand_dims(self.mean, 0), 0)
        std = np.expand_dims(np.expand_dims(self.std, 0), 0)
        images = ((images / 255.) - mean) / std
        labels = np.array(labels).astype(np.float32)

        images = torch.from_numpy(images).float()
        # B H W 3 ->B 3 H W
        images = images.permute(0, 3, 1, 2).contiguous()
        labels = torch.from_numpy(labels).long()

        return {
            'image': images,
            'label': labels,
        }


class Opencv2PIL:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = Image.fromarray(np.uint8(image))

        return {
            'image': image,
            'label': label
        }


class PIL2Opencv:
    def __init__(self):
        pass

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = np.asarray(image).astype(np.float32)

        return {
            'image': image,
            'label': label
        }


class TorchRandomResizedCrop:
    def __init__(self, resize=224):
        self.resize_func = torchvision.transforms.RandomResizedCrop(size=int(resize))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.resize_func(image)

        return {
            'image': image,
            'label': label
        }


class TorchRandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.func = torchvision.transforms.RandomHorizontalFlip(p=prob)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.func(image)

        return {
            'image': image,
            'label': label
        }


class TorchRandomCrop:

    def __init__(self, resize=224):
        self.RandomCrop = torchvision.transforms.RandomCrop(int(resize))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.RandomCrop(image)

        return {
            'image': image,
            'label': label,
        }


class TorchResize:

    def __init__(self, resize=224):
        self.Resize = torchvision.transforms.Resize(int(resize))

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = self.Resize(image)

        return {
            'image': image,
            'label': label,
        }


if __name__ == "__main__":
    from util import get_paths

    dataset_path = 'D:\\workspace\\data\\dl\\flower'
    train_images_path, train_images_label = get_paths(dataset_path, "train", 'flower_indices.json')

    transforms = {
        'train': torchvision.transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=224),
            TorchRandomHorizontalFlip(prob=0.5),
            PIL2Opencv()
        ])
    }
    cls_collater = ClassificationCollater(mean=[0.5, 0.5, 0.5],
                                          std=[0.5, 0.5, 0.5])

    train_dataset = CustomDataset(train_images_path, train_images_label, transforms['train'])

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset,
                              batch_size=12,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True,
                              collate_fn=cls_collater)

    from tqdm import tqdm
    for datas in tqdm(train_loader):
        images = datas['image']
        labels = datas['label']
        print(images.shape, labels.shape)
    




