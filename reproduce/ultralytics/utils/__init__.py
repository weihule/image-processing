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
from typing import Union
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
    level = logging.INFO if verbose and RANK in {-1, 0} else logging.ERROR

    # 用utf8配置stdout输出
    formatter = logging.Formatter("%(message)s")
    if WINDOWS and hasattr(sys.stdout, "encoding") and sys.stdout.encoding != "utf-8":
        class CustomFormatter(logging.Formatter):
            def format(self, record):
                """Sets up logging with UTF-8 encoding and configurable verbosity."""
                return emojis(super().format(record))
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            elif hasattr(sys.stdout, "buffer"):
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            else:
                formatter = CustomFormatter("%(message)s")
        except Exception as e:
            print(f"Creating custom formatter for non UTF-8 environments due to {e}")
            formatter = CustomFormatter("%(message)s")
    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    # Set up the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


# Set Logger
LOGGER = set_logging(LOGGING_NAME, verbose=VERBOSE)

def emojis(string=""):
    return string.encode().decode("ascii", "ignore") if WINDOWS else string


def yaml_save(file="data.yaml", data=None, header=""):
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
    
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)
    
    # 输出data到yaml文件（errors="ignore": 编码错误时忽略，防止崩溃）
    with open(file, "w", errors="ignore", encoding="utf-8") as fw:
        if header:
            fw.write(header)
        # allow_unicode=True: 中文等Unicode字符正常显示
        yaml.safe_dump(data, fw, sort_keys=False, allow_unicode=True)


def yaml_load(file="data.yaml", append_filename=False):
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, "r", errors="ignore", encoding="utf-8") as fr:
        s = fr.read()  # string

        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data
    

def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=float("inf"))
    LOGGER.info(f"Printing '{colorstr('bold', 'green', yaml_file)}'\n\n{dump}")
    

# 默认配置
DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
DEFAULT_SOL_DICT = yaml_load(DEFAULT_SOL_CFG_PATH) 
for k, v in DEFAULT_CFG_DICT.items():
    print(f"k = {k} v = {v}")

# from ultralytics.utils.patches import torch_load, torch_save, imread, imwrite, imshow
# torch.load = torch_load
# torch.save = torch_save
# if WINDOWS:
#     # Apply cv2 patches for non-ASCII and non-UTF characters in image paths
#     cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow

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

    set_logging(name="ultralytics", verbose=True)
    logger = logging.getLogger("ultralytics")
    logger.info("This is an info message")

    content = yaml_print(yaml_file=r'D:\workspace\code\image-processing\reproduce\ultralytics\cfg\datasets\coco.yaml')
    print(content)


if __name__ == "__main__":
    # test()
    pass


