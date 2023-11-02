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


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)



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
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    weight_path = r"/home/8TDISK/weihule/training_data/seg/unet_1.0/checkpoint_epoch5.pth"
    state_dict = torch.load(weight_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)
    model.to(device)
    filepath = r"/home/8TDISK/weihule/data/test_images/car2.jpg"
    # filepath = r"/home/8TDISK/weihule/data/car/img/0cdf5b5d0ce1_02.jpg"
    img = Image.open(filepath)
    mask = predict_img(model=model,
                       full_image=img,
                       device=device)

    result = mask_to_image(mask, mask_values)
    result.save("./res.png")
    # plot_img_and_mask(img, mask)


if __name__ == "__main__":
    run()

