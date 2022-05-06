import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def read_split_data(root: str):
    train_images_path = list()  # 存储训练集的所有图片路径
    train_images_label = list()  # 存储训练集图片对应索引信息
    val_images_path = list()  # 存储验证集的所有图片路径
    val_images_label = list()  # 存储验证集图片对应索引信息
    modes = ['train', 'val']
    for mode in modes:
        for curdir, dirs, files in os.walk(os.path.join(root, mode)):
            for file in files:
                fn_path = os.path.join(curdir, files)
            
         

if __name__ == "__main__":
    # root = "D:\\PythonTemp\\data\\DL\\flower"
    # read_split_data(root)
    test_dict = {"a":65, "b":66, "c":67, "d":68}
