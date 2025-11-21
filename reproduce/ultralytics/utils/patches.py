
from __future__ import annotations

import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import Any
from packaging import version

import cv2
import numpy as np
import torch

def imread(filename: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    file_bytes = np.fromfile(filename, np.uint8)
    if filename.endswith((".tiff", ".tif")):
        success, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
        if success:
            # Handle RGB images in tif/tiff format
            return frames[0] if len(frames) == 1 and frames[0].ndim == 3 else np.stack(frames, axis=2)
        return None
    else:
        im = cv2.imdecode(file_bytes, flags)
        return im[..., None] if im is not None and im.ndim == 2 else im  # Always ensure 3 dimensions
    

def imwrite(filename: str, img: np.ndarray, params: list[int] | None = None) -> bool:
    try:
        cv2.imencode(Path(filename).suffix, img, params)[1].tofile(filename)
        return True
    except Exception:
        return False
    

def imshow(winname: str, mat: np.ndarray) -> None:
    cv2.imshow(winname.encode('unicode_escape').decode(), mat)


def torch_load(*args, **kwargs):
    torch_version = str(torch.__version__)

    if version.Version(torch_version) >= version.Version("1.13.0") and "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    return torch.load(*args, **kwargs)

def torch_save(*args, **kwargs):
    for i in range(4):
        try:
            return torch.save(*args, **kwargs)
        except RuntimeError as e:
            if i == 3:
                raise e
            time.sleep((2**i) / 2)  # 0.5s, 1.0s, 2.0s