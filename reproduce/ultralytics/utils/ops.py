from __future__ import annotations

import contextlib
import math
import re
import time
import cv2
import numpy as np

import torch
import torch.nn.functional as F

class Profile(contextlib.ContextDecorator):
    def __init__(self, t: float = 0.0, device: torch.device | None = None):
        """
        Initialize the Profile class.

        Args:
            t (float): Initial accumulated time in seconds.
            device (torch.device, optional): Device used for model inference to enable CUDA synchronization.
        """
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))
    def __enter__(self):
        """Start timing."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):  # noqa
        """Stop timing."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def __str__(self):
        """Return a human-readable string representing the accumulated elapsed time."""
        return f"Elapsed time is {self.t} s"

    def time(self):
        """Get current time with CUDA synchronization if applicable."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()
    

def segment2box(segment, width: int = 640, height: int = 640):
    """
    Args:
        segment (torch.Tensor): Segment coordinates in format (N, 2) where N is number of points.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.

    Returns:
        (np.ndarray): Bounding box coordinates in xyxy format [x1, y1, x2, y2].
    """
    if isinstance(segment, torch.Tensor):
        segment = segment.cpu().numpy()
    dtype = segment.dtype  

    x, y = segment.T  # segment xy

    # Clip coordinates if 3 out of 4 sides are outside the image
    if np.array([x.min()<0, y.min()<0, x.max()>width, y.max()>height]) >= 3:
        x = x.clip(0, width)
        y = y.clip(0, height)
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x = x[inside]
    y = y[inside]
    return (
        np.array([x.min(), y.min(), x.max(), y.max()], dtype=dtype)
        if any(x)
        else np.zeros(4, dtype=dtype)
    )


def clip_boxes(boxes, shape):
    """
    Clip bounding boxes to image boundaries.

    Args:
        boxes (torch.Tensor | np.ndarray): Bounding boxes to clip.
        shape (tuple): Image shape as HWC or HW (supports both).

    Returns:
        (torch.Tensor | np.ndarray): Clipped bounding boxes.
    """
    h, w = shape[:2]
    if isinstance(boxes, torch.Tensor):
        # 原地修改
        boxes[..., 0].clamp_(0, w)
        boxes[..., 1].clamp_(0, h)
        boxes[..., 2].clamp_(0, w)
        boxes[..., 3].clamp_(0, h)
    else:
        boxes[..., 0] = boxes[..., 0].clip(0, w)
        boxes[..., 1] = boxes[..., 1].clip(0, h)
        boxes[..., 2] = boxes[..., 2].clip(0, w)
        boxes[..., 3] = boxes[..., 3].clip(0, h)
    return boxes

def clip_coords(coords, shape):
    """
    Args:
        coords (torch.tensor | np.ndarray): Line coordinates to clip.
        shape (tuple): Image shape as HWC or HW (supports both).
    """
    h, w = shape[:2]
    if isinstance(coords, torch.Tensor):
        coords[..., 0] = coords[..., 0].clamp(0, w)
        coords[..., 1] = coords[..., 0].clamp(0, h)
    else:
        coords[..., 0] = coords[..., 0].clip(0, w)
        coords[..., 1] = coords[..., 0].clip(0, h)
    return coords


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """
    Args:
        img1_shape (tuple): shape of the source image (h, w).
        boxes (torch.tensor): bounding boxes to rescale in format (N, 4).
        img0_shape (tuple): shape of the target image (h, w).
        ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling.
        padding (bool): Whether box are based on YOLO-style augmented images with padding.
        xywh (bool): Whether box format is xywh (True) or xyxy (False).
    """
    # calculate from img0_shape
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])    # gain = old / new
        pad_x = round((img1_shape[1] - img0_shape[1]*gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]
    
    if padding:
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        if not xywh:
            boxes[..., 2] -= pad_x
            boxes[..., 3] -= pad_y
    boxes[..., 4] /= gain
    return clip_boxes(boxes, img0_shape)


def make_divisible(x: int, divisor):
    """
    Args: 
        x (int): The number to make divisible.
        divisor (int | torch.Tensor): Ther divisor.
    """
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def scale_imge(masks, im0_shape, ratio_pad=None):
    """
    Args:
        masks (np.ndarray): Resize and padded maskes with shape [H, W, N] or [H, W, 3]
        im0_shape (tuple): Original image shape as HWC or HW (support both).
        ratio_pad (tuple, optional): Ratio and padding values as ((ratio_h, ratio_w), (pad_h, pad_w)).
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im0_h, im0_w = im0_shape[:2]
    im1_h, im1_w, _ = masks.shape
    if im1_h == im1_h and im1_w == im0_w:
        return masks
    
    if ratio_pad is None:   # calculate from im0_shape
        gain = min(im1_h / im0_h, im1_w / im0_w)
        pad = (im1_w - im0_w * gain) / 2, (im1_h - im0_h * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]

    pad_w, pad_h = pad
    top = int(round(pad_h - 0.1))
    left = int(round(pad_w - 0.1))
    bottom = im1_h - int(round(pad_h + 0.1))
    right = im1_w - int(round(pad_w + 0.1))

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_w, im0_h))
