import random
import numpy as np


def binary_func(nums, find_num, simple_loc=True):
    """
    nums: 有序列表
    find_num: 需要查找的数字
    simple_loc: 为True, 返回查找数字的单个位置
                为False, 返回查找数字的所有位置
    """
    if len(nums) == 0:
        return -1
    start, end = 0, len(nums)-1
    if simple_loc:
        while start <= end:
            mid = start + (end - start) // 2
            if find_num == nums[mid]:
                return mid
            elif find_num < nums[mid]:
                end = mid - 1 
            else:
                start = mid + 1
        
        # 如果没有查找到, 返回 -1
        return -1
    else:
        while start <= end:
            mid = start + (end - start) // 2
            if find_num == nums[mid]:
                locs = [mid]
                mid_origin = mid

                # 往右走
                mid = mid_origin + 1
                while 0<=mid<=len(nums)-1 and nums[mid] == find_num:
                    locs.append(mid)
                    mid += 1
                
                # 往左走
                mid = mid_origin - 1
                while 0<=mid<=len(nums)-1 and nums[mid] == find_num:
                    locs.append(mid)
                    mid -= 1
                return locs
            elif find_num < nums[mid]:
                end = mid -1 
            else:
                start = mid + 1
        # 如果没有查找到, 返回 -1
        return -1


def bubble_sort(arr, reverse=False):
    """
    reverse为True, 从大到小排序
    reverse为False, 从小到大排序
    """
    if reverse:
        for i in range(0, len(arr)-1):
            for j in range(0, len(arr)-1-i):
                if arr[j] <= arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
    else:
        for i in range(0, len(arr)-1):
            for j in range(0, len(arr)-1-i):
                if arr[j] >= arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


if __name__ == "__main__":

    arr = list(np.random.randint(1, 15, size=10))
    print(arr)
    arr_sort = bubble_sort(arr)
    rand_num = random.sample(arr_sort, 1)
    print(arr_sort)
    print(rand_num)
    loc = binary_func(arr_sort, rand_num)
    locs = binary_func(arr_sort, rand_num, False)
    print('loc = ', loc)
    print('loc = ', locs)