import os
import json
import logging
from logging import handlers
import torch
import json


def get_paths(root, mode, classes_json_file):
    with open(classes_json_file, 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
        class2index = infos['classes']

    imgs_path, labels_path = [], []
    for folder in os.listdir(os.path.join(root, mode)):
        for fn in os.listdir(os.path.join(root, mode, folder)):
            img_path = os.path.join(root, mode, folder, fn)
            label = class2index[folder]
            imgs_path.append(img_path)
            labels_path.append(label)

    return imgs_path, labels_path


def get_logger(name, log_dir):
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.log'.format(name))

    # 按 D(天) 为单位来分割日志
    info_handler = handlers.TimedRotatingFileHandler(filename=info_name,
                                                     when='D',
                                                     encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def load_state_dict(saved_model_path, model, excluded_layer_name):
    if saved_model_path is None:
        print('No pretrained model file !')
        return
    save_state_dict = torch.load(saved_model_path, map_location='cpu')
    filtered_state_dict = {name: weight for name, weight in save_state_dict.items()
                           if name in model.state_dict() and weight.shape == model.state_dict()[name].shape
                           and not (name in excluded_layer_name)}
    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load !')
    else:
        print(f'loading {len(filtered_state_dict)} layers parameters !')
        model.load_state_dict(filtered_state_dict, strict=False)


def get_indices(root, save_path):
    infos = {}
    for idx, fn in enumerate(os.listdir(root)):
        infos[idx] = fn
    infos = {v: k for k, v in infos.items()}
    infos = {'classes': infos}

    json_str = json.dumps(infos, indent=4, ensure_ascii=False)
    with open(save_path, "w") as json_file:
        json_file.write(json_str)


if __name__ == "__main__":
    get_indices('/root/autodl-tmp/imagenet100/imagenet100_val',
                '/code/study/torch_classification/utils/imagenet100.json')


