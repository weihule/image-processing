import torch
import numpy as np
import time
from shapely.geometry import Polygon




def iou(bbox1, bbox2):
    inter = (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])) * \
            (min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    print(inter)


if __name__ == "__main__":
    start = time.time()
    # arr1 = [1, 1, 4, 1, 4, 6, 1, 6]
    # arr2 = [2, 3, 7, 3, 7, 9, 2, 9]
    # area1 = Polygon(np.array(arr1).reshape(4, 2))
    # area2 = Polygon(np.array(arr2).reshape(4, 2))
    # inter = area1.intersection(area2).area
    # print(area1, area2, inter)

    ar1 = [1, 1, 4, 6]
    ar2 = [2, 3, 7, 9]
    iou(ar1, ar2)
    end = start - time.time()
    print(f'running time: {end-start}')