import torch
import numpy as np
from shapely.geometry import Polygon

arr1 = [1, 1, 4, 1, 4, 6, 1, 6]
arr2 = [2, 3, 7, 3, 7, 9, 2, 9]
area1 = Polygon(np.array(arr1).reshape(4, 2))
area2 = Polygon(np.array(arr2).reshape(4, 2))
inter = area1.intersection(area2).area
print(area1, area2, inter)

ar1 = [1, 1, 4, 6]
ar1 = [1, 1, 4, 6]
def iou(bbox1, bbox2):
    pass
