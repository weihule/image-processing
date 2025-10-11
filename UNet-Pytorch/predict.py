import os
from PIL import Image
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


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


def main():
    model_path = r""
    classes = [0, 1]
    cfgs = {
        "model_path": model_path,
        "classes": classes
    }
    