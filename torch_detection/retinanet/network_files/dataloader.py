import os
import sys
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage

from PIL import Image

def main(a, b):
    try:
        res = a/b
    except (ValueError, ZeroDivisionError) as e:
        print(e)


if __name__ == "__main__":
    main(2, 0)
