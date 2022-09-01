import os
import torch
from torchvision import transforms
import backbones

from utils.datasets import FlowerDataset, ClassificationCollater, CustomDataset
from utils.datasets import Opencv2PIL, PIL2Opencv, TorchRandomHorizontalFlip, TorchResize, TorchRandomResizedCrop
from utils.util import get_paths


class Config:
    mode = 'local'      # company   autodl

    if mode == 'local':
        dataset_path = 'D:\\workspace\\data\\dl\\flower'
        pre_weight_path = 'D:\\workspace\\data\\classification_data\\yolov4backbone\\pths\\yolov4cspdarknet53-acc77.448.pth'
        save_root = 'D:\\workspace\\data\\classification_data\\yolov4backbone'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        resume = os.path.join(checkpoints, 'latest.pth')
    elif mode == 'company':
        dataset_path = '/workshop/weihule/data/dl/flower'
        pre_weight_path = '/workshop/weihule/data/weights/yolov4backbone/darknet530.835.pth'
        # pre_weight_path = None
        save_path = '/workshop/weihule/data/weights/yolov4backbone/darknet53'
        save_root = '/workshop/weihule/data/weights/yolov4backbone'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        resume = os.path.join(checkpoints, 'latest.pth')
    elif mode == 'company2':
        dataset_path = '/workspace/mingqianshi/workspace/data/dl/flower'
        pre_weight_path = '/workspace/mingqianshi/workspace/data/weight/yolov4backbone/yolov4cspdarknet53-acc77.448.pth'
        save_root = '/workspace/mingqianshi/workspace/data/weight/yolov4backbone'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        resume = os.path.join(checkpoints, 'latest.pth')
    else:
        dataset_path = '/root/autodl-tmp/flower'
        pre_weight_path = '/root/autodl-nas/classification_data/yolov4backbone/yolov4cspdarknet53-acc77.448.pth'
        save_root = '/root/autodl-nas/classification_data/yolov4backbone'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        resume = os.path.join(checkpoints, 'latest.pth')

    epochs = 50
    batch_size = 24
    lr = 0.001
    lrf = 0.001
    num_workers = 4
    freeze_layer = False
    apex = True
    seed = 1
    print_interval = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms = {
        'train': transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=224),
            TorchRandomHorizontalFlip(prob=0.5),
            PIL2Opencv()
        ]),
        'val': transforms.Compose([
            Opencv2PIL(),
            TorchRandomResizedCrop(resize=224),
            PIL2Opencv()
        ])
    }

    train_images_path, train_images_label = get_paths(dataset_path, "train", 'utils/flower_indices.json')
    val_images_path, val_images_label = get_paths(dataset_path, "val", 'utils/flower_indices.json')

    train_dataset = CustomDataset(train_images_path, train_images_label, transforms['train'])
    val_dataset = CustomDataset(val_images_path, val_images_label, transforms['val'])

    backbone_type = 'yolov4_csp_darknet53'
    model = backbones.__dict__[backbone_type](
        **{'num_classes': 5,
           'act_type': 'leakyrelu'}
    )

    mean = [0.485, 0.456, 0.406],
    std = [0.229, 0.224, 0.225]
    cls_collater = ClassificationCollater(mean=mean,
                                          std=std)


if __name__ == "__main__":
    dataset_path = 'D:\\workspace\\data\\dl\\flower'
    train_images_path, train_images_label = get_paths(dataset_path, "train", 'utils/flower_indices.json')
    for i, j in zip(train_images_path, train_images_label):
        print(i, j)


