import json
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def coco91_to_coco80_class():
    """
    Converts 91-index COCO class IDs to 80-index COCO class IDs.

    Returns:
        (list): A list of 91 class IDs where the index represents the 80-index class ID and the value is the
            corresponding 91-index class ID.
    """
    return [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        None,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        None,
        24,
        25,
        None,
        None,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        None,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        None,
        60,
        None,
        None,
        61,
        None,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        None,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        None,
    ]


def coco80_to_coco91_class():
    r"""
    Converts 80-index (val2014) to 91-index (paper).
    For details see https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/.

    Example:
        ```python
        import numpy as np

        a = np.loadtxt("data/coco.names", dtype="str", delimiter="\n")
        b = np.loadtxt("data/coco_paper.names", dtype="str", delimiter="\n")
        x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
        x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
        ```
    """
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


def min_index(arr1, arr2):
    """
    arr1 (np.ndarray): [N, 2]
    arr1 (np.ndarray): [M, 2]
    """
    # [N, 1, 2] - [1, M, 2] -> sum -> [N, M]
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    # axis=None 表示在整个数组中找最小值, 返回的是线性索引
    min_1d = np.argmin(dis, axis=None)
    # 线性索引 -> 二维索引 (例: 5 (shape: 3*3) -> [1, 2])
    min_2d = np.unravel_index(min_1d, dis.shape)
    return min_2d


def merge_multi_segment(segments):
    """
    把多个分离的分段(segments)通过找距离最近的点连接,合成一个完整的轮廓
    segments = [
        [10, 20, 15, 25, 20, 30],           # 分段1：3个点 (10,20), (15,25), (20,30)
        [21, 31, 25, 35, 30, 40],           # 分段2：3个点 (21,31), (25,35), (30,40)
        [31, 41, 35, 45, 40, 50]            # 分段3：3个点 (31,41), (35,45), (40,50)
    ]
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    for i in range(1, len(segments)):
        # print(i)
        idx1, idx2 = min_index(segments[i-1], segments[i])
        # print(f"idx1={idx1} idx2={idx2}")
        idx_list[i-1].append(idx1)
        idx_list[i].append(idx2)
    """
    idx_list: [[2], [0, 2], [0]] 
    seg0的索引为2的点（连接seg0和seg1的）
    seg1的索引为0（连接seg0和seg1的）和索引为2的点（连接seg1和seg2的）
    seg2的索引为0的点（连接seg1和seg2的）
    """
    
    """
    第一轮：正向处理
    - 处理分段的前向部分
    - 建立连接点

    第二轮：反向处理
    - 添加中间分段中间被分割的部分
    - 确保完整性

    原因：中间的分段被分成两部分
    - 前半部分（第一轮）：从连接点A到连接点B
    - 后半部分（第二轮）：从连接点B到末尾，再从开头到连接点A
    """
    for k in range(2):
        # 前向连接
        if k == 0:
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # Deal with the first segment and the last one
                if i in {0, len(idx_list) - 1}:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in {0, len(idx_list) - 1}:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def test():
    segments = [
        [10, 20, 15, 25, 20, 30],           
        [21, 31, 25, 35, 30, 40],          
        [31, 41, 35, 45, 40, 50] 
    ]
    ret = merge_multi_segment(segments)
    print(f"ret = {ret} {len(ret)}")


if __name__ == "__main__":
    test()



    
