import gc
import math
import os
import random
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_KEYS,
    LOGGER,
    NUM_THREADS,
    PYTHON_VERSION,
    TORCHVISION_VERSION,
    WINDOWS,
    __version__,
    colorstr
)
from ultralytics.utils.checks import 