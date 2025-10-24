import os
from PIL import Image
from loguru import logger
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet


def preprocess(pil_img, scale):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.BICUBIC)
    img = np.asarray(pil_img)
    img = img.transpose((2, 0, 1))
    if (img > 1).any():
        img = img / 255.0
    
    return img


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eavl()
    img = preprocess(full_img, scale_factor)
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()


def mask2image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)


def plot_img_and_mask(img, mask):
    print(f"mask = {mask}")
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes+1, figsize=(5 * (classes + 1), 5))
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    ax[0].axis('off')  # 隐藏坐标轴框
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
        ax[i + 1].axis('off')
    plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    model_path = r""
    classes = 2
    cfgs = {
        "model_path": model_path,
        "classes": classes,
        "scale": 0.5,
        "mask-threshold": 0.5,
        "visualize": True,
        "output_dir": ""
    }


def test():
    img_path = r'D:\workspace\data\images\carvana-image-masking-challenge\train\0cdf5b5d0ce1_01.jpg'
    mask_path = r'D:\workspace\data\images\carvana-image-masking-challenge\train_masks\0cdf5b5d0ce1_01_mask.gif'
    plot_img_and_mask(Image.open(img_path), np.array(Image.open(mask_path)))


if __name__ == "__main__":
    test()




    