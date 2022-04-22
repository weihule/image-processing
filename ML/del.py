import os
from re import sub
import numpy as np
import lmdb
import operator
from math import log

# arr = np.arange(9).reshape((3, 3))
# print(arr)
# np.random.shuffle(arr)
# arr_1 = np.random.permutation(arr)
# print(arr)
# print(arr_1)

# a = np.array((1, 2, 5, 2, 0.1))
# print(np.argmin(a))

# print(np.zeros((3, 1)))

# arr = []
# for i in range(24):
#     arr.append(np.random.randint(1, 10))
# arr = np.array(arr).reshape((12, 2))
# print(arr)
# mean_ = np.mean(arr, axis=1)
# sum_ = np.sum(arr, axis=1)
# print(mean_, mean_.shape, mean_.flatten().shape, type(mean_))
# print(sum_, sum_.shape, type(sum_))

# print(np.empty((10, 1)))

# a = np.array([[[0], [1], [0], [0], [2], [0], [1]]])
# print(a == 1)
# print((a == 0).flatten())

# a = np.zeros((10, 3))
# print(a)
# a[4] = 1
# a[3][0] = 1
# print(a)

# a = np.array([
#     [1,10,1],
#     [2,20,2],
#     [3,30,3],
#     [2,40,4]
# ])

# a = np.arange(12).reshape((3, 4))
# print(a)
# sub = np.delete(a, 1, axis=0)
# print(sub, sub.shape)

# arr = np.array([[[1, 2, 3, 4], 
#                 [5, 6, 7, 8],
#                 [9, 10, 11, 12]],
#                 [[13, 14, 15, 16], 
#                 [17, 18, 19, 20],
#                 [21, 22, 23, 24]]])

# print(arr.shape)

# print(arr[..., :2], arr[..., :2].shape)

# print(arr[..., 2], arr[..., 2].shape)


# a = np.array([
#     [1,10,1],
#     [2,20,2],
#     [3,30,3],
#     [2,40,4]
# ])

# arr = np.arange(12) + 1
# mask = np.ones(len(arr), dtype=bool)

# mask[[0, 2, 4]] = False
# # result = arr[mask,...]
# result = np.delete(arr, [0, 2, 4], axis=0)
# print(result)

# print(a[0])
# print(len(a[0]))
# print(a[:, 1])

# mask = a[:, 0] == 2
# print(mask)
# a[mask, 1] = 100
# print(a)

# c = {"a":4, "b":1, "c":10, "d":8, "e":3}
# res = sorted(c.items(), key=operator.itemgetter(1))
# print(res)

# a = [2, 12, 52, 50]
# b = operator.itemgetter(1)
# print(b(a))

# c = operator.itemgetter(1, 0, 3)
# print(c(a))

# a=[('john','A',15), ('jane','B',12), ('dave','B',10), ('judy','C',12)]
# res = sorted(a, key=operator.itemgetter(2, 1))
# print(res)

# a = (-3/5*log(3/5, 2)-2/5*log(2/5, 2))*(10/15)
# b = (-1/5*log(1/5, 2)-4/5*log(4/5, 2))*(5/15)

# print(a + b)

# a = (-6/9*log(6/9, 2)-3/9*log(3/9, 2))*(9/15)
# b = 0

# print(a + b)


dataset = np.array([[0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']])

def splitdata(dataset, index, feature_value):
    mask = dataset[:, index] == feature_value
    subdataset = dataset[mask]
    print(subdataset)
    subdataset = np.delete(subdataset, index, axis=1)
    print(subdataset)

if __name__ == "__main__":
    splitdata(dataset, 2, "0")





