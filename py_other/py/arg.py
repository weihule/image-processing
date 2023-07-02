import os
import argparse
import json
import sys
from pathlib import Path
import os
from tqdm import tqdm
from loguru import logger

sink = sys.stdout

logger.add(
    sink='./service.log',
    rotation='500 MB',                  # 日志文件最大限制500mb
    retention='30 days',                # 最长保留30天
    format="{time}|{level}|{message}",  # 日志显示格式
    compression="zip",                  # 压缩形式保存
    encoding='utf-8',                   # 编码
    level='DEBUG',                      # 日志级别
    enqueue=True,                       # 默认是线程安全的，enqueue=True使得多进程安全
)

logger.debug("详细调试信息")
logger.info("普通信息")
logger.success("成功信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.trace("异常信息")
logger.critical("严重错误信息")


@logger.catch
def my_function(x, y, z):
    # An error? It's caught anyway!
    return 1 / (x + y + z)


my_function(0, 0, 0)


def main():
    print("sys.argv[0] = ", sys.argv[0])
    print("__file__ = ", __file__)

    arr1 = sys.argv[1]
    arr2 = sys.argv[2]
    print(arr1)
    print(arr2)


class DataSet(object):
    @property
    def method_with_property(self):
        return 15

    def method_without_property(self):
        return 15


def test():
    d = DataSet()
    print(d.method_with_property)
    print(d.method_without_property())


def mkdir_if_missing(file_path):
    p = Path(file_path)
    if not p.exists():
        p.mkdir(parents=True)


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


if __name__ == "__main__":
    # main()
    test()

