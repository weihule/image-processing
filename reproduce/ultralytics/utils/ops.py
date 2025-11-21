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

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    if ratio_pad is None:
        pass

