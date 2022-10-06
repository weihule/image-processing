import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F
from utils import GradCAM, show_cam_on_image

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__))))
)
from study.torch_reid.mine_reid.torchreid import models


def main(config):
    gpu_devices = config['gpu_devices']
    save_dir = config['save_dir']
    dataset = config['dataset']
    root = config['root']
    split_id = config['split_id']
    cuhk03_labeled = config['cuhk03_labeled']
    cuhk03_classic_split = config['cuhk03_classic_split']
    use_metric_cuhk03 = config['use_metric_cuhk03']
    arch = config['arch']
    loss_type = config['loss_type']
    height = config['height']
    width = config['width']
    test_batch = config['test_batch']
    pth_path = config['pth_path']
    aligned = config['aligned']
    reranking = config['reranking']
    test_distance = config['test_distance']

    with open('../torch_reid/mine_reid/torchreid/datas/datasets/images/market1501.json', 'r') as fr:
        pid2label = json.load(fr)

    model = models.init_model(name=arch,
                              num_classes=751,
                              loss=loss_type,
                              aligned=aligned,
                              act_func='prelu',
                              attention=None)
    # model.eval()
    # arr = torch.randn(1, 3, 224, 224)
    # outs = model(arr)
    # print(outs.shape)
    target_layers = [model.conv5]
    data_transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # load image
    img_path = "D:\\workspace\\data\\dl\\market1501\\Market-1501-v15.09.15\\bounding_box_train\\0002_c1s1_000451_03.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')

    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    input_tensor = torch.unsqueeze(img_tensor, dim=0)

    # model.eval()
    # outs = model(input_tensor)
    # outs = F.log_softmax(outs, dim=-1)
    # print(outs.shape)
    # prob, class_index = torch.max(outs, dim=-1)
    # print(prob, class_index)

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    target_category = pid2label[str(int(img_path.split(os.sep)[-1].split('_')[0]))]
    # target_category = 48
    print(target_category)

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]
    img = np.array(img, dtype=np.uint8)

    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.show()


if __name__ == '__main__':
    configs = {
        'gpu_devices': '0',
        'save_dir': '',
        'dataset': 'market1501',
        'root': 'D:\\workspace\\data\\dl',
        'split_id': 0,
        'cuhk03_labeled': False,
        'cuhk03_classic_split': False,
        'use_metric_cuhk03': False,
        'arch': 'osnet_x1_0',
        'loss_type': 'grad_cam',
        'height': 256,
        'width': 128,
        'test_batch': 32,
        'pth_path': 'D:\\workspace\\data\\reid_data\\osnet_softmax_epoch50_9273\\softmax_epoch50.pth',
        'aligned': False,
        'reranking': True,
        'test_distance': 'global'
    }
    main(configs)
