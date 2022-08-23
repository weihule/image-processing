import os
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

