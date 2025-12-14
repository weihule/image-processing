import contextlib
import importlib.metadata
import inspect
import json
import logging
import os
import platform
import re
import socket
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from urllib.parse import unquote

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original


# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

# Other Constants
ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]      # 向上定位两级，定位到ultralytics这个目录下
ASSETS = ROOT / "assets"
DEFAULT_CFG_PATH = ROOT / "cfg/default.yaml"
DEFAULT_SOL_CFG_PATH = ROOT / "cfg/solutions/default.yaml" 
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))
AUTOINSTALL = str(os.getenv("YOLO_AUTOINSTALL", True)).lower() == 'true'
VERBOSE = str(os.getenv("YOLO_VERBOSE", True)).lower() == "true"
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}" if VERBOSE else None 
LOGGING_NAME = "ultralytics"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
ARM64 = platform.machine() in {"arm64", "aarch64"} 
PYTHON_VERSION = platform.python_version()
TORCH_VERSION = torch.__version__
TORCHVISION_VERSION = importlib.metadata.version("torchvision")  # faster than importing torchvision
IS_VSCODE = os.environ.get("TERM_PROGRAM", False) == "vscode"

# 环境变量
torch.set_printoptions(linewidth=320, precision=4, profile="default")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for deterministic training to avoid CUDA warning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs



def colorstr(*input):
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

class TQDM(tqdm_original):
    def __init__(self, *args, **kwargs):
        # 全局变量优先, 当VERBOSE是False时（默认是True），不管传入的diable是什么，该值都是True
        kwargs["disable"] = not VERBOSE or kwargs.get("disable", False)
        kwargs.setdefault("bar_format", TQDM_BAR_FORMAT)
        # 传递所有参数给原生 tqdm
        super().__init__(*args, **kwargs)

class SimpleClass:
    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith("_"):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f"{a}: {v.__module__}.{v.__class__.__name__} object"
                else:
                    s = f"{a}: {repr(v)}"
                attr.append(s)
        return f"{self.__module__}.{self.__class__.__name__} object with attributes:\n\n" + "\n".join(attr)

    def __repr__(self):
        return self.__str__()
    
    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


class IterableSimpleNameSpace(SimpleNamespace):
    def __iter__(self):
        # vars(self) 等价于 self.__dict__，返回属性字典
        return iter(vars(self).items())
    
    def __str__(self):
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())
    
    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' object has no attribute '{attr}'. This may be caused by a modified or out of date ultralytics
            'default.yaml' file.\nPlease update your code with 'pip install -U ultralytics' and if necessary replace
            {DEFAULT_CFG_PATH} with the latest version from
            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
            """
        )

    def get(self, key, default=None):
        """Return the value of the specified key if it exists; otherwise, return the default value."""
        return getattr(self, key, default)
    

def plt_settings(rcparams=None, backend="Agg"):
    if rcparams is None:
        rcparams = {"font.size": 11}
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_backend = plt.get_backend()
            switch = backend.lower() != original_backend.lower()
            if switch:
                plt.close("all")
                plt.switch_backend(backend)
            try:
                with plt.rc_context(rcparams):
                    result = func(*args, **kwargs)
            finally:
                if switch:
                    plt.close("all")
                    plt.switch_backend(original_backend)
            return result
        return wrapper
    return decorator


def set_logging(name="LOGGING_NAME", verbose=True):
    

from ultralytics.utils.patches import torch_load, torch_save, imread, imwrite, imshow
torch.load = torch_load
torch.save = torch_save
if WINDOWS:
    # Apply cv2 patches for non-ASCII and non-UTF characters in image paths
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow

def test():
    plat = platform.machine()
    system = platform.system()
    print(f"plat = {plat} system = {system}") 
    print(f"FILE = {FILE}")
    print(f"ROOT = {ROOT}")
    print(Path(__file__).resolve().parents[1])
    ret1, ret2, ret3 = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
    print(f"{ret1, ret2, ret3}")

    # for i in TQDM(range(100)):
    #     print(i)


if __name__ == "__main__":
    test()


