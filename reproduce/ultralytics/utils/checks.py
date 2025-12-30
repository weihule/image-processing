import glob
import inspect
import math
import os
import platform
import re
import shutil
import subprocess
import time
from importlib import metadata
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import requests

import torch

from ultralytics.utils import (
    ASSETS,
    AUTOINSTALL,
    IS_COLAB,
    IS_GIT_DIR,
    IS_KAGGLE,
    IS_PIP_PACKAGE,
    LINUX,
    LOGGER,
    MACOS,
    ONLINE,
    PYTHON_VERSION,
    ROOT,
    TORCHVISION_VERSION,
    USER_CONFIG_DIR,
    WINDOWS,
    Retry,
    SimpleNamespace,
    ThreadingLocked,
    TryExcept,
    clean_url,
    colorstr,
    emojis,
    url2file
)

def parse_version(version="0.0.0") -> tuple:
    try:
        return tuple(map(int, re.findall(r"\d+", version)[:3]))
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ failure for parse_version({version}), returning (0, 0, 0): {e}")
        return 0,0,0

def check_version(
        current: str = "0.0.0",
        required: str = "0.0.0",
        name: str = "version",
        hard: bool = False,
        verbose: bool = False,
        msg: str = "",
) -> bool:
    """
    Examples:
        # Check if current version is exactly 22.04
        check_version(current="22.04", required="==22.04")

        # Check if current version is greater than or equal to 22.04
        check_version(current="22.10", required="22.04")

        # Check if current version is less than or equal to 22.04
        check_version(current="22.04", required="<=22.04")

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current="21.10", required=">20.04,<22.04")
    """

    if not current: # if current is '' or None
        LOGGER.warning(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():
        try:
            name = current
            current = metadata.version(current)
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(emojis(f"WARNING ⚠️ {current} package is required but not installed")) from e
            else:
                return False
    
    if not required:    # if required is '' or None
        return True
    
    if 'sys_platform' in required and (    # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
    ):
        return True
    
    op = ""
    version = ""
    result = True
    c = parse_version(current)  # '1.2.3' -> (1, 2, 3)










