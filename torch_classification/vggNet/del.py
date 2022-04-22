import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import random

# weights = nn.Conv2d(4, 8, kernel_size=2, groups=4)
# print(weights.weight.shape)
# img = Image.open("./daisy01.jpg")
a = np.random.random((1, 8))
a = torch.tensor(a)
res = torch.argmax(a, dim=-1)
print(res)