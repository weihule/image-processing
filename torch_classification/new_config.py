import os
import torch
from torchvision import transforms

import backbones
from datasets.imagenet100 import ImageNet100Dataset
from datasets.transform import *


class Config:
    mode = 'local'  # company   autodl

    if mode == 'local':
        # dataset_path = 'D:\\workspace\\data\\dl\\flower'
        # pre_weight_path = 'D:\\workspace\\data\\classification_data\\yolov4backbone\\pths\\yolov4cspdarknet53-acc77.448.pth'
        # save_root = 'D:\\workspace\\data\\classification_data\\yolov4backbone'
        # log = os.path.join(save_root, 'log')
        # checkpoints = os.path.join(save_root, 'checkpoints')
        # resume = os.path.join(checkpoints, 'latest.pth')

        dataset_path = 'D:\\workspace\\data\\dl\\imagenet100'
        pre_weight_path = "D:\\Desktop\\resnet50-acc76.264.pth"
        save_root = 'E:\\workspace\\classification\\resnet50'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        pth_path = os.path.join(save_root, 'pths')
        resume = os.path.join(checkpoints, 'latest.pth')

    elif mode == 'company1':
        dataset_path = '/workshop/weihule/data/dl/flower'
        # pre_weight_path = '/workshop/weihule/data/weights/yolov4backbone/darknet530.835.pth'
        pre_weight_path = None
        save_root = '/workshop/weihule/data/classification_data/mobilenet'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        pth_path = os.path.join(save_root, 'pths')
        resume = os.path.join(checkpoints, 'latest.pth')
    elif mode == 'company2':
        dataset_path = '/workspace/mingqianshi/workspace/data/dl/flower'
        pre_weight_path = '/workspace/mingqianshi/workspace/data/weight/yolov4backbone/yolov4cspdarknet53-acc77.448.pth'
        save_root = '/workspace/mingqianshi/workspace/data/weight/yolov4backbone'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        pth_path = os.path.join(save_root, 'pths')
        resume = os.path.join(checkpoints, 'latest.pth')
    else:
        # dataset_path = '/root/autodl-tmp/flower'
        # pre_weight_path = '/root/autodl-nas/classification_data/resnet/pths/resnet50-acc76.322.pth'
        # save_root = '/root/autodl-nas/classification_data/resnet'
        # log = os.path.join(save_root, 'log')
        # checkpoints = os.path.join(save_root, 'checkpoints')
        # pth_path = os.path.join(save_root, 'pths')
        # resume = os.path.join(checkpoints, 'latest.pth')

        dataset_path = '/root/autodl-tmp/imagenet100'
        pre_weight_path = '/root/autodl-nas/classification_data/resnet/pths/resnet50_imagenet100_acc56.pth'
        save_root = '/root/autodl-nas/classification_data/resnet'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        pth_path = os.path.join(save_root, 'pths')
        resume = os.path.join(checkpoints, 'latest.pth')

    epochs = 80
    batch_size = 8
    lr = 0.001
    lrf = 0.001
    num_workers = 4
    freeze_layer = False
    apex = False
    seed = 0
    print_interval = 50
    save_interval = 1
    image_size = 224
    num_classes = 100
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_dict = {
        'train': transforms.Compose([
            OpenCV2PIL(),
            TorchResize(resize=image_size),
            TorchRandomHorizontalFlip(prob=0.5),
            TorchColorJitter(),
            PIL2OpenCV(),
            Random2DErasing()
        ]),
        'val': transforms.Compose([
            OpenCV2PIL(),
            TorchResize(resize=image_size),
            PIL2OpenCV()
        ])
    }

    train_dataset = ImageNet100Dataset(root_dir=dataset_path,
                                       set_name="imagenet100_train",
                                       transform=transform_dict["train"])
    val_dataset = ImageNet100Dataset(root_dir=dataset_path,
                                     set_name="imagenet100_val",
                                     transform=transform_dict["val"])

    backbone_type = 'resnet50'
    model = backbones.__dict__[backbone_type](
        **{'num_classes': num_classes}
    )

    collater = Collater(mean=mean,
                        std=std)


cfg = Config()


if __name__ == "__main__":
    print(len(Config.train_dataset))
