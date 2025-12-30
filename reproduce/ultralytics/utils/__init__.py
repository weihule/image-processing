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
    if isinstance(v, str) and v.lower() == "none":
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNameSpace(**DEFAULT_CFG_DICT)

def read_device_model() -> str:
    """
    存储嵌入式设备（如树莓派、Jetson 开发板）的硬件型号信息。
    """
    try:
        with open("/proc/device-tree/model") as f:
            return f.read()
    except Exception as e:
        return ""
    
def is_ubuntu() -> bool:
    try:
        with open("/etc/os-release") as f:
            return "ID=ubuntu" in f.read()
    except FileNotFoundError:
        return False
    
def is_colab():
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ

def is_kaggle():
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"

def is_docker():
    try:
        with open("/proc/self/cgroup") as f:
            return "docker" in f.read()
    except Exception:
        return False
    
def is_raspberrypi() -> bool:
    return "Raspberry Pi" in PROC_DEVICE_MODEL

def is_jetson() -> bool:
    return any(keyword in PROC_DEVICE_MODEL.lower() for keyword in ("nvidia", "jetson"))

def is_online() -> bool:
    try:
        assert str(os.getenv("YOLO_OFFLINE", "")).lower() != "true"
        import socket

        for dns in ("1.1.1.1", "8.8.8.8"):
            socket.create_connection(address=(dns, 80), timeout=2.0).close()
            return True
    except Exception:
        return False
    
def is_pip_package(filepath: str = __name__) -> bool:
    """判断指定的名称是否是一个通过 pip 安装的、非内置的 Python 包 / 模块"""
    import importlib.util
    spec = importlib.util.find_spec(filepath)
    return spec is not None and spec.origin is not None

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    return os.access(str(dir_path), os.W_OK)

def is_pytest_running():
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules) or ("pytest" in Path(ARGV[0]).stem)

def is_github_action_running() -> bool:
    """
    Determine if the current environment is a GitHub Actions runner.

    Returns:
        (bool): True if the current environment is a GitHub Actions runner, False otherwise.
    """
    return "GITHUB_ACTIONS" in os.environ and "GITHUB_WORKFLOW" in os.environ and "RUNNER_OS" in os.environ


def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory. If
    the current file is not part of a git repository, returns None.

    Returns:
        (Path | None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / ".git").is_dir():
            return d


def is_git_dir():
    """
    Determines whether the current file is part of a git repository. If the current file is not part of a git
    repository, returns None.

    Returns:
        (bool): True if current file is part of a git repository.
    """
    return GIT_DIR is not None


def get_git_origin_url():
    """
    Retrieves the origin URL of a git repository.

    Returns:
        (str | None): The origin URL of the git repository or None if not git directory.
    """
    if IS_GIT_DIR:
        try:
            origin = subprocess.check_output(["git", "config", "--get", "remote.origin.url"])
            return origin.decode().strip()
        except subprocess.CalledProcessError:
            return None


def get_git_branch():
    """
    Returns the current git branch name. If not in a git repository, returns None.

    Returns:
        (str | None): The current git branch name or None if not a git directory.
    """
    if IS_GIT_DIR:
        try:
            origin = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            return origin.decode().strip()
        except subprocess.CalledProcessError:
            return None


def get_default_args(func):
    """
    Returns a dictionary of default arguments for a function.

    Args:
        func (callable): The function to inspect.

    Returns:
        (dict): A dictionary where each key is a parameter name, and each value is the default value of that parameter.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_ubuntu_version():
    """
    Retrieve the Ubuntu version if the OS is Ubuntu.

    Returns:
        (str): Ubuntu version or None if not an Ubuntu OS.
    """
    if is_ubuntu():
        try:
            with open("/etc/os-release") as f:
                return re.search(r'VERSION_ID="(\d+\.\d+)"', f.read())[1]
        except (FileNotFoundError, AttributeError):
            return None


def get_user_config_dir(sub_dir="Ultralytics"):
    """
    Return the appropriate config directory based on the environment operating system.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        (Path): The path to the user config directory.
    """
    if WINDOWS:
        path = Path.home() / "AppData" / "Roaming" / sub_dir
    elif MACOS:  # macOS
        path = Path.home() / "Library" / "Application Support" / sub_dir
    elif LINUX:
        path = Path.home() / ".config" / sub_dir
    else:
        raise ValueError(f"Unsupported operating system: {platform.system()}")

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(path.parent):
        LOGGER.warning(
            f"WARNING ⚠️ user config directory '{path}' is not writeable, defaulting to '/tmp' or CWD."
            "Alternatively you can define a YOLO_CONFIG_DIR environment variable for this path."
        )
        path = Path("/tmp") / sub_dir if is_dir_writeable("/tmp") else Path().cwd() / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path

# Define constants (required below)
PROC_DEVICE_MODEL = read_device_model()   # is_jetson() and is_raspberrypi() depend on this constant
ONLINE = is_online()
IS_COLAB = is_colab()
IS_KAGGLE = is_kaggle()
IS_DOCKER = is_docker()
IS_JETSON = is_jetson()
IS_PIP_PACKAGE = is_pip_package()
IS_RASPBERRYPI = is_raspberrypi()
GIT_DIR = get_git_dir()
IS_GIT_DIR = is_git_dir()
USER_CONFIG_DIR = Path(os.getenv("YOLO_CONFIG_DIR") or get_user_config_dir())  # Ultralytics settings dir
SETTINGS_FILE = USER_CONFIG_DIR / "settings.json"


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

def removw_colorstr(input_string):
    """
    Examples:
        >>> remove_colorstr(colorstr("blue", "bold", "hello world"))
        >>> "hello world"
    """
    ansi_escape = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
    return ansi_escape.sub("", input_string)


class TryExcept(contextlib.ContextDecorator):
    """
    Examples:
        As a decorator:
        >>> @TryExcept(msg="Error occurred in func", verbose=True)
        >>> def func():
        >>> # Function logic here
        >>>     pass

        As a context manager:
        >>> with TryExcept(msg="Error occurred in block", verbose=True):
        >>> # Code block here
        >>>     pass
    """
    def __init__(self, msg="", verbose=True):
        self.msg = msg
        self.verbose = verbose
    
    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))


class Retry(contextlib.ContextDecorator):
    """
    Examples:
        >>> @Retry(times=3, delay=2)
        >>> def test_func():
        >>> # Replace with function logic that may raise exceptions
        >>>     return True
    """
    def __init__(self, times=3, delay=2):
        self.times = times
        self.delay = delay
        self._attemps = 0
    
    def __call__(self, func):
        def wrapped_func(*args, **kwargs):
            self._attemps = 0
            while self._attemps < self.times:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._attemps += 1
                    print(f"Retry {self._attemps}/{self.times} failed: {e}")
                    if self._attemps >= self.times:
                        raise e
                    time.sleep(self.delay * (2**self._attempts))
        return wrapped_func
    

def threaded(func):
    """
    除非传入参数'threaded=False'，否则该函数会在单独的线程中运行。
    """
    def wrapper(*args, **kwargs):
        # 如果不传 "threaded" 参数，返回True，即默认func在新线程运行
        if kwargs.pop("threaded", True):
            thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
            thread.start()
            return thread
        else:
            return func(*args, **kwargs)
    return wrapper


class ThreadingLocked:
    """
    Examples:
        from ultralytics.utils import ThreadingLocked

        @ThreadingLocked()
        def my_function():
            # Your code here
    """
    def __init__(self):
        self.lock = threading.Lock()

    def __call__(self, f):
        from functools import wraps

        @wraps(f)
        def decorated(*args, **kwargs):
            with self.lock:
                return f(*args, **kwargs)
        return decorated



class JSONDict(dict):
    """
    Examples:
        >>> json_dict = JSONDict("data.json")
        >>> json_dict["key"] = "value"
        >>> print(json_dict["key"])
        value
        >>> del json_dict["key"]
        >>> json_dict.update({"new_key": "new_value"})
        >>> json_dict.clear()
    """
    def __init__(self, file_path: Union[str, Path] = "data.json"):
        super().__init__()
        self.file_path = file_path
        self.lock = Lock()
        self._load()

    def _load(self):
        try:
            if self.file_path.exists():
                with open(self.file_path) as f:
                    self.update(json.load(f))
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {self.file_path}. Starting with an empty dictionary.")
        except Exception as e:
            print(f"Error reading from {self.file_path}: {e}")

    def _save(self):
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as fw:
                json.dump(dict(self), fw, indent=2, default=self._json_default)
        except Exception as e:
            print(f"Error writing to {self.file_path}: {e}")
    
    @staticmethod
    def _json_default(obj):
        """
        JSON 标准只支持这些基本类型：字符串、数字、布尔值、null、数组、对象
        当 json.dump() 遇到不能直接序列化的对象（比如 Path 对象、datetime 等），就会报错,
        所以这个函数就用来处理Path这个对象
        """
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    def __setitem__(self, key, value):
        with self.lock:
            # 等价于 d["name"] = "Alice"
            super().__setitem__(key, value)
            self._save()

    def __str__(self):
        contents = json.dumps(dict(self), indent=2, ensure_ascii=False, default=self._json_default)
        return f'JSONDict("{self.file_path}"):\n{contents}'

    def update(self, *args, **kwargs):
        """Update the dictionary and persist changes."""
        with self.lock:
            super().update(*args, **kwargs)
            self._save()

    def clear(self):
        """Clear all entries and update the persistent storage."""
        with self.lock:
            super().clear()
            self._save()


class SettingManager(JSONDict):
    def __init__(self, file=SETTINGS_FILE, version="0.0.6"):
        import hashlib


def deprecation_warn(arg, new_arg=None):
    msg = f"WARNING ⚠️ '{arg}' is deprecated and will be removed in in the future."
    if new_arg is not None:
        msg += f" Use '{new_arg}' instead."
    LOGGER.warning(msg)


def clean_url(url):
    """Strip auth from URL, i.e. https://url.com/file.txt?auth -> https://url.com/file.txt."""
    url = Path(url).as_posix().replace(":/", "://")
    return unquote(url).split("?")[0]

def url2file(url):
    """Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt."""
    return Path(clean_url(url)).name


def vscode_msg(ext="ultralytics.ultralytics-snippets") -> str:
    """Display a message to install Ultralytics-Snippets for VS Code if not already installed."""
    path = (USER_CONFIG_DIR.parents[2] if WINDOWS else USER_CONFIG_DIR.parents[1]) / ".vscode/extensions"
    obs_file = path / ".obsolete"  # file tracks uninstalled extensions, while source directory remains
    installed = any(path.glob(f"{ext}*")) and ext not in (obs_file.read_text("utf-8") if obs_file.exists() else "")
    url = "https://docs.ultralytics.com/integrations/vscode"
    return "" if installed else f"{colorstr('VS Code:')} view Ultralytics VS Code Extension ⚡ at {url}"


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


