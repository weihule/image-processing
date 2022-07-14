# 双指针问题


"""
给你一个 升序排列 的数组 nums ，
请你 原地 删除重复出现的元素，
使每个元素 只出现一次 ，
返回删除后数组的新长度。
元素的 相对顺序 应该保持 一致 。
"""
from shutil import move


def main1(arr):
    fast, slow = 1, 0
    while fast < len(arr):
        if arr[fast] == arr[slow]:
            fast += 1
        else:
            slow += 1
            arr[slow] = arr[fast]
            fast += 1
    
    return arr[:slow+1]


# 买卖股票的最佳时机
def main2(test2):
    days = len(test2)

    # d_profit_0 = 0
    # d_profit_1 = -test2[0]
    # print(d_profit_0, d_profit_1)
    # for i in range(1, days-1):
    #     d_profit_0 = max(d_profit_0, d_profit_1 + test2[i])
    #     d_profit_1 = max(d_profit_1, d_profit_0 - test2[i])
    #     print(d_profit_0, d_profit_1)
    # res = max(d_profit_0, d_profit_1 + test2[days-1])

    res = 0
    for d in range(days-1):
        if test2[d+1] > test2[d]:
            res += (test2[d+1] - test2[d])
        else:
            continue

    return res


# 旋转数组
def main3(nums, k):
    # 1.先全部反转
    nums = nums[::-1]

    # 2.反转前k个
    nums[:k] = nums[:k][::-1]

    # 3.反转剩余的
    nums[k-len(nums):] = nums[k-len(nums):][::-1]


# 删除零
def move_zero(nums):
    index = 0
    for j in range(len(nums)):
        if nums[j] == 0:
            continue
        else:
            nums[index] = nums[j]
            index += 1
    while(index < len(nums)):
        nums[index] = 0
        index += 1

    return nums



if __name__ == "__main__":
    # arr = [1, 1, 2, 3, 3, 4, 5]
    # arr = [1, 1, 2]

    # print(arr)
    # res = main1(arr)

    # test2 = [7, 1, 5, 3, 6, 4]
    # res = main2(test2)
    # print(res)

    # nums = [1, 2, 3, 4, 5, 6, 7]
    # nums_set = set(nums)
    # if nums_set.add(8):
    #     print('this is ')
    # print(nums_set)

    # k = 3
    # main3(nums, k)
    # print(nums)

    nums = [0,1,0,3,12]
    res = move_zero(nums)
    print(res)
