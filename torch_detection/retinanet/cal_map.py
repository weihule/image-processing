import torch
import numpy as np
from shapely.geometry import Polygon

arr1 = [1, 1, 4, 1, 4, 6, 1, 6]
area1 = Polygon(np.array(arr1).reshape(4, 2)).area
print(area1)

