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
    
)

def check_version(
        current: str = "0.0.0",
        resuired: str = "0.0.0",
        name: str = "version",
        hard: bool = False,
        verbose: bool = False,
        msg: str = "",
) -> bool:
    if not current: # if current is '' or None
        