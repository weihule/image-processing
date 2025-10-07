import os
import torch
from torchvision import transforms

from backbones import init_model
from datasets.transform import *
from datasets.collater import Collater
from datasets.data_manager import init_dataset


class Config:
    mode = 'local'  # company   autodl

    if mode == 'local':
        # dataset_path = 'D:\\workspace\\data\\dl\\flower'
        # pre_weight_path = 'D:\\workspace\\data\\classification_data\\yolov4backbone\\pths\\yolov4cspdarknet53-acc77.448.pth'
        # save_root = 'D:\\workspace\\data\\classification_data\\yolov4backbone'
        # log = os.path.join(save_root, 'log')
        # checkpoints = os.path.join(save_root, 'checkpoints')
        # resume = os.path.join(checkpoints, 'latest.pth')

        dataset_path = 'D:\\workspace\\data\\dl\\flower'
        pre_weight_path = r"D:\workspace\data\training_data\resnet\resnet50-acc76.264.pth"
        save_root = r'D:\workspace\data\training_data\resnet50\resnet50'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        pth_path = os.path.join(save_root, 'pths')
        resume = os.path.join(checkpoints, 'resume.pth')

    elif mode == 'autodl':
        dataset_path = '/root/autodl-tmp/imagenet100'
        pre_weight_path = '/root/autodl-nas/classification_data/resnet/pths/resnet50_imagenet100_acc56.pth'
        save_root = '/root/autodl-nas/classification_data/resnet'
        log = os.path.join(save_root, 'log')
        checkpoints = os.path.join(save_root, 'checkpoints')
        pth_path = os.path.join(save_root, 'pths')
        resume = os.path.join(checkpoints, 'latest.pth')
    else:
        dataset_path = None
        pre_weight_path = None
        save_root = None
        log = None
        checkpoints = None
        pth_path = None
        resume = None

    seed = 0

    # model
    backbone_type = 'resnet50'
    num_classes = 100
    model = init_model(backbone_type=backbone_type,
                       num_classes=num_classes)

    # data
    dataset_name = "flower5"
    epochs = 15
    batch_size = 8
    lr = 0.001
    lrf = 0.001
    num_workers = 4
    freeze_layer = False
    apex = False
    print_interval = 50
    save_interval = 1
    image_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_dataset = init_dataset(name=dataset_name,
                                 root_dir=dataset_path,
                                 set_name="train",
                                 class_file=cfgs[cfgs["mode"]]["class_file"],
                                 transform=transform["train"])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfgs["batch_size"],
                              shuffle=True,
                              num_workers=cfgs["num_workers"],
                              collate_fn=collater)

    val_dataset = init_dataset(name=cfgs["dataset_name"],
                               root_dir=cfgs[cfgs["mode"]]["root_dir"],
                               set_name=cfgs["val_set_name"],
                               class_file=cfgs[cfgs["mode"]]["class_file"],
                               transform=transform["val"])
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=cfgs["batch_size"],
                            shuffle=True,
                            num_workers=cfgs["num_workers"],
                            collate_fn=collater)

    # optimizer scheduler
    step_size = 10
    gamma = 0.1
    T_max = epochs

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

    collater = Collater(mean=mean,
                        std=std)


cfg = Config()


def main():
    import yaml
    with open("cfg.yaml", "r", encoding="utf-8") as fr:
        cfgs = yaml.load(fr, Loader=yaml.FullLoader)
    print(cfgs)


if __name__ == "__main__":
    main()

