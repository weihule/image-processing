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

def segments2boxes(segments):
    """
    输入示例: [[N, 2], [M, 2]]
    即多个轮廓的xy点集列表，每个子元素都是numpy类型
    """

    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    return xyxy2xywh(np.array(boxes))  # cls, xywh

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
    if isinstance(coords, torch.Tensor):
        coords[..., 0] = coords[..., 0].clamp(0, shape[1])
        coords[..., 1] = coords[..., 1].clamp(0, shape[0])
    else:
        coords[..., 0] = coords[..., 0].clip(0, shape[1])
        coords[..., 1] = coords[..., 1].clip(0, shape[0])
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


def scale_imgage(masks, im0_shape, ratio_pad=None):
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


def xyxy2xywh(x):
    """
    xyxy  = x1, y1, x2, y2                                  (左上和右下)
    xywh  = x_center, y_center, width, height               (中心点坐标和宽高)
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = (x1 + x2) / 2
    y[..., 1] = (y1 + y2) / 2
    y[..., 2] = x2 - x1
    y[..., 3] = y2 - y1
    return y


def xywh2xyxy(x):
    """
    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x, y, width, height) format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def xywhn2xyxy(x, w: int = 640, h: int = 640, padw: int = 0, padh: int = 0):
    """
    Args:
        x (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, w, h) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        padw (int): Padding width in pixels.
        padh (int): Padding height in pixels.

    Returns:
        y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x) 
    xc, yc, xw, xh = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    half_w, half_h = xw / 2, xh / 2
    # 归一化坐标是基于缩放后未填充的图像计算的，
    # 而我们最终要得到的是填充后完整图像上的像素坐标，所以需要把填充的偏移量加回去。
    y[..., 0] = w * (xc - half_w) + padw
    y[..., 1] = h * (yc - half_h) + padh
    y[..., 2] = w * (xc + half_w) + padw
    y[..., 3] = h * (yc + half_h) + padh
    return y


def xyxy2xywhn(x, w: int = 640, h: int = 640, clip: bool = False, eps: float = 0.0):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): Image width in pixels.
        h (int): Image height in pixels.
        clip (bool): Whether to clip boxes to image boundaries.
        eps (float): Minimum value for box width and height.

    Returns:
        (np.ndarray | torch.Tensor): Normalized bounding box coordinates in (x, y, width, height) format.
    """
    if clip:
        x = clip_boxes(x, (h-eps, w-eps))
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x) 
    x1, y1, x2, y2 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    y[..., 0] = ((x1 + x2) / 2) /  w
    y[..., 1] = ((y1 + y2) / 2) /  h
    y[..., 2] = (x2 - x1) /  w
    y[..., 3] = (y2 - y1) /  h
    return y


def xywh2ltwh(x):
    """
    xywh  = x_center, y_center, width, height      (中心坐标)
            
    ltwh  = left, top, width, height               (左上角坐标)
            └─────┬─────┘
            左上角坐标
    Args:
        x (np.ndarray | torch.Tensor): Input bounding box coordinates in xywh format.

    Returns:
        (np.ndarray | torch.Tensor): Bounding box coordinates in xyltwh format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    return y


def xyxy2ltwh(x):
    """
    xyxy  = x1, y1, x2, y2      (左上和右下坐标)
            
    ltwh  = left, top, width, height               (左上角坐标)
            └─────┬─────┘
            左上角坐标
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y


def ltwh2xywh(x):
    """     
    ltwh  = left, top, width, height               (左上角坐标)
            └─────┬─────┘
            左上角坐标
    xywh  = x_center, y_center, w, h      (中心点坐标和宽高)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 2] / 2
    y[..., 1] = x[..., 1] + x[..., 3] / 2 
    return y


def ltwh2xyxy(x):
    """     
    ltwh  = left, top, width, height               (左上角坐标)
            └─────┬─────┘
            左上角坐标
    xyxy  = x1, y1, x2, y2      (左上和右下坐标)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]  # width
    y[..., 3] = x[..., 3] + x[..., 1]  # height
    return y


def xyxyxyxy2xywhr(x):
    """
    Convert batched Oriented Bounding Boxes (OBB) from [xy1, xy2, xy3, xy4] to [xywh, rotation] format.
    xyxyxyxy  = x1y2, x2y2, x3y3, x4y4               (四点坐标)
    xywhr = cx, cy, w, h, r      (中心点坐标, 宽高， 弧度)
    """
    is_torch = isinstance(x, torch.Tensor)
    points = x.cpu().numpy() if is_torch else x
    points = points.reshape(len(x), -1, 2)
    rboxes = []
    for pts in points:
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        rboxes.append([cx, cy, w, h, angle / 180 * np.pi])
    return torch.tensor(rboxes, device=x.device, dtype=x.dtype) if is_torch else np.asarray(rboxes)


def xywhr2xyxyxyxy(x):
    cos, sin, cat, stack = (
        (torch.cos, torch.sin, torch.cat, torch.stack)
        if isinstance(x, torch.Tensor)
        else (np.cos, np.sin, np.concatenate, np.stack)
    )
    ctr = x[..., :2]    # [N, 2]
    w, h, angle = (x[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = cat(vec1, -1)    # [N, 2]
    vec2 = cat(vec2, -1)    # [N, 2]
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return stack([pt1, pt2, pt3, pt4], -2)


def resample_segments(segments, n=1000):
    """
    输入一个列表，列表里的每个元素是[n, 2]的numpy数组
    """
    for i, s in enumerate(segments):
        if len(s) == n:
            continue
        s = np.concatenate((s, s[0:1, :]), axis=0) # 把起点追加到末尾
        x = np.linspace(0, len(s)-1, n-len(s) if len(s) < n else n)
        xp = np.arange(len(s))
        x = np.insert(x, np.searchsorted(x, xp), xp) if len(s) < n else x
        segments[i] = (
            np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)], dtype=np.float32).reshape(2, -1).T
        )  # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None, normalize=False, padding=True):
    """
    把分割坐标(x,y)从img1缩放到img0（从处理后的图像空间转回到原始图像空间）
    img1_shape[0] = img0_shape[0] * gain + 2*pad[1]  # 高度
    img1_shape[1] = img0_shape[1] * gain + 2*pad[0]  # 宽度
    """
    if ratio_pad is None: 
        gain = min(img1_shape[0]/img0_shape[0],img1_shape[1]/img0_shape[1])     # gain = old / new
        pad = (img1_shape[1]-img0_shape[1]*gain) / 2, (img1_shape[0]-img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    if padding:
        coords[..., 0] -= pad[0]    # x padding
        coords[..., 1] -= pad[1]    # y padding
    coords[..., 0] /= gain
    coords[..., 1] /= gain
    coords = clip_coords(coords, img0_shape)
    if normalize:
        coords[..., 0] /= img0_shape[1]     # width
        coords[..., 1] /= img0_shape[0]     # height
    return coords


def regularize_rboxes(rboxes):
    """
    规范化旋转框，确保旋转角度在[0, pi/2]
    Args:
        rboxes (torch.Tensor): Input boxes of shape(N, 5) in xywhr format.
    Returns:
        (torch.Tensor): The regularized boxes.
    """
    x, y, w, h, t = rboxes.unbind(dim=-1)

    w_ = torch.where(w > h, w, h)
    # 如果 w > h，保持 w；否则用 h 代替
    # 结果：w_ = max(w, h)

    h_ = torch.where(w > h, h, w)
    # 如果 w > h，保持 h；否则用 w 代替
    # 结果：h_ = min(w, h)

    t = torch.where(w > h, t, t + math.pi / 2) % math.pi
    # 如果原本 w > h：保持角度 t
    # 如果原本 w <= h：加上 π/2（因为交换了宽高）
    # 然后对 π 取模，确保在 [0, π) 范围内
    
    # 重新堆叠
    return torch.stack([x, y, w_, h_, t], dim=-1)


def convert_torch2numpy_batch(batch: torch.Tensor) -> np.ndarray:
    """
    把一个 FP32 的batch tensor (0.0-1.0) 转成 np.uint8 (0-255)
    BCHW -> BHWC
    """
    batch = (batch.permute(0, 2, 3, 1).contiguous()*255).clamp(0, 255)
    batch = batch.to(torch.uint8).cpu().numpy()
    return batch


def masks2segments(masks, strategy="all"):
    """
    将批量的二值掩码（mask）张量，转换为对应的轮廓线段（segment）列表
    mask (torch.Tensor): 
    """
    from ultralytics.data.converter import merge_multi_segment
    segments = []
    for x in masks.int.cpu().numpy().astype(np.uint8):
        
        """
        cv2.findContours(...) 会提取出 2 个外轮廓，因此 c 是长度为 2 的列表；
        c[0]：第一个正方形的轮廓坐标，形状为 (4, 1, 2)（4 个顶点，每个顶点是 (x,y)）；
        c[1]：第二个正方形的轮廓坐标，形状也为 (4, 1, 2)；
        代码中 strategy="all" 时，会把 c[0] 和 c[1] 合并为一个 (8, 2) 的数组；
        代码中 strategy="largest" 时，会选 c[0]/c[1] 中点数更多的那个（若点数相同，选第一个）。
        OpenCV 返回的 (N, 1, 2) 有冗余维度，因此代码中用 reshape(-1, 2) 把它简化为 (N, 2)：
        """
        c = cv2.findContours(x, x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            # 合并所有分割目标
            if strategy == "all": 
                if len(c) > 1: 
                    c = np.concatenate(
                        merge_multi_segment([x.reshape(-1, 2) for x in c])
                        )
                else:  
                    c = c[0].reshape(-1, 2)
            # 选择点数更多的目标
            elif strategy == "largest":
                c = np.array(
                        c[np.array([len(x) for x in c]).argmax()]
                    ).reshape(-1, 2)
        else:
            c = np.zeros(shape=(0, 2))  # 无分割目标
        segments.append(c.astype(np.float32))
    return segments

def clear_str(s):
    """
    例如：输入"hello@world!123+abc€def", 输出"hello_world_123_abc_def"
    """
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]",
                  repl="_",
                  string=s)

    

def empty_like(x):
    ret = torch.empty_like(x, dtype=torch.float32) if isinstance(x, torch.Tensor) else np.empty_like(x, dtype=np.float32)
    return ret


def test():
    arr = np.array([[0, 0], [0, 10], [20, 10], [20, 0]], dtype=np.float32)
    print(arr.shape)
    arr2 = resample_segments([arr], n=8)
    print(arr2[0], arr2[0].shape)


if __name__ == "__main__":
    test()

