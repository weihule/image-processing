import os
import cv2
import torch
import numpy as np
import json
import torch.nn.functional as F
from backbones.yolov4backbone import yolov4_csp_darknet53, yolov4_csp_darknet_tiny


def main(cfgs):
    with open('utils/flower_indices.json', 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
        class2index = infos['classes']
    index2class = {v: k for k, v in class2index.items()}

    model = cfgs['net']
    img_root = cfgs['img_root']
    model_path = cfgs['model_path']
    batch_size = cfgs['batch_infer']
    size = cfgs['size']
    device = cfgs['device']
    mean = cfgs['mean']
    std = cfgs['std']

    mean = np.expand_dims(np.expand_dims(mean, 0), 0)
    std = np.expand_dims(np.expand_dims(std, 0), 0)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    imgs_path = []
    for folder in os.listdir(img_root):
        for fn in os.listdir(os.path.join(img_root, folder)):
            fn_path = os.path.join(img_root, folder, fn)
            imgs_path.append(fn_path)
    split_imgs = [imgs_path[p:p+batch_size] for p in range(0, len(imgs_path), batch_size)]

    with torch.no_grad():
        for sub_lists in split_imgs:
            images = []
            images_name = []
            for fn in sub_lists:
                images_name.append(fn.split(os.sep)[-1])
                img = cv2.imread(fn)
                img = cv2.resize(img, (size, size))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.
                img = img - mean / std
                images.append(img)
            images_tensor = [torch.tensor(p, dtype=torch.float32) for p in images]
            images_tensor = torch.stack(images_tensor, dim=0).to(device)
            images_tensor = images_tensor.permute(0, 3, 1, 2).contiguous()
            preds = model(images_tensor)
            preds = F.softmax(preds, dim=1)
            probs, indices = torch.max(preds, dim=1)
            probs, indices = probs.cpu().numpy().tolist(), indices.cpu().numpy().tolist()
            for img_name, prob, idx in zip(images_name, probs, indices):
                print(img_name, round(prob, 3), index2class[idx])


if __name__ == "__main__":
    mode = 'company'
    if mode == 'local':
        imgs = ''
        model_path = ''
    elif mode == 'company':
        imgs = '/workshop/weihule/data/dl/flower/test'
        model_path = '/workshop/weihule/data/weights/yolov4backbone/darknet53_857.pth'
    else:
        imgs = '/workshop/weihule/data/dl/flower/test'
        model_path = ''

    net = yolov4_csp_darknet53(num_classes=5)
    cfg_dict = {
        'img_root': imgs,
        'model_path': model_path,
        'net': net,
        'batch_infer': 8,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'size': 256,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }

    main(cfg_dict)




