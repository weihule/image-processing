import os
import random
import numpy as np

# 参考资料
"https://www.cnblogs.com/wuxinyan/p/8615127.html"
"https://www.cnblogs.com/Mufasa/p/10527387.html"

def bubble(arr):
    """
    冒泡排序
    平均时间复杂度 O(n**2)
    """
    for i in range(0, len(arr)-1):
        for j in range(0, len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selection(arr):
    """
    选择排序
    """
    for i in range(0, len(arr)-1):
        # 记录最小值的索引
        minIndex = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[minIndex]:
                minIndex = j
        if i != minIndex:
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
    return arr

def insertion(arr):
    """
    插入排序
    平均时间复杂度 O(n**2)
    """
    for i in range(len(arr)):
        pre_index = i-1       
        current = arr[i]
        # 需要注意的是，这里是从pre_index倒数往前进行遍历的
        while pre_index >= 0 and arr[pre_index] > current:    
            arr[pre_index+1] = arr[pre_index]
            pre_index -= 1
        arr[pre_index+1] = current
    return arr


def insertion2(arr):
    for i in range(len(arr)):
        for j in range(i, 0, -1):
            if arr[j] < arr[j-1]:
                arr[j], arr[j-1] =  arr[j-1], arr[j]
            else:
                break
    return arr


if __name__ == "__main__":
    random.seed(0)
    arr = [5, 12, 11, 10, 6, 8, 0, 5, 9]
    print(arr)

    # arr_new = bubble(arr)
    # arr_new = selection(arr)
    arr_new = insertion2(arr)
    print(arr_new)
    # print(sorted(arr))

    

