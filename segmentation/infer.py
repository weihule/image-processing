import os
from loguru import logger
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from models import UNet


def plot_img_and_mask(image, mask):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes+1)
    ax[0].set_title('Input image')
    ax[0].imshow(image)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([])
    plt.yticks([])
    plt.show()
