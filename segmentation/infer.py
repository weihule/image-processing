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
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(image)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def image_process(pil_img, scale):
    w, h = pil_img.size
    new_w, new_h = int(scale * w), int(scale * h)
    assert new_w > 0 and new_h > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
    img = np.asarray(pil_img)
    img = img.transpose((2, 0, 1))
    img = img / 255.
    return img


def predict_img(model, full_image, device, scale_factor=1, out_threshold=0.5):
    model.eval()
    print(f"full_image.size = {full_image.size}")
    img = image_process(full_image, scale_factor)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    print(f"img.shape = {img.shape}")

    with torch.no_grad():
        outputs = model(img).cpu()
        output = F.interpolate(outputs, (full_image.size[1], full_image.size[0]), mode='bilinear')
        if model.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def run():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet(n_channels=3, n_classes=2, bilinear=True)
    a1 = list(model.state_dict().keys())
    weight_path = r"D:\Desktop\unet.pth"
    state_dict = torch.load(weight_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    a2 = list(state_dict.keys())
    print(len(a1), len(a2))
    model.load_state_dict(state_dict)
    # filepath = r"D:\workspace\data\dl\car\img\0cdf5b5d0ce1_02.jpg"
    # img = Image.open(filepath)
    # mask = predict_img(model=model,
    #                    full_image=img,
    #                    device=device)
    # plot_img_and_mask(img, mask)


if __name__ == "__main__":
    run()
