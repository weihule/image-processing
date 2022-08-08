
"""
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
"""
def pa_lou(n):
    hash_dict = dict()
    hash_dict[1] = 1
    hash_dict[2] = 2
    # if n == 2:
    #     return 2
    # elif n == 1:
    #     return 1
    # else:
    for k, v in hash_dict.items():
        print(k, v)
        
    if n in hash_dict:
        return hash_dict[n]
    else:
        hash_dict[n] = pa_lou(n-1) + pa_lou(n-2)
        return hash_dict[n]


"""
买卖股票的最佳时机
"""
def maxProfit(prices) -> int:
    min_val_idx = 0         # 记录最小值的索引
    max_profit = 0          # 记录最大利润
    for idx, val in enumerate(prices):
        if prices[min_val_idx] > val:
            min_val_idx = idx
        max_profit = max(val-prices[min_val_idx], max_profit)
    return max_profit


# 最大自序和
def maxSubArray(nums) -> int:
    if len(nums) == 0:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    # 这里错了
    # max_value = 0


    # ====================
    # d = []
    # d.append(nums[0])
    # max_value = d[0]
    # for i in range(1, len(nums)):
    #     curr = max(d[i-1], 0) + nums[i]
    #     d.append(curr)
    #     max_value = max(max_value, d[i])
    # return max_value
    # ====================

    # ====================
    # 代码优化
    max_value = nums[0]
    for i in range(1, len(nums)):
        curr = max(max_value, 0) + nums[i]
        max_value = max(curr, max_value)
        print(max_value)
    return max_value



if __name__ == "__main__":
    # n = 4
    # res = pa_lou(n)
    # print('res = ', res)

    # arr = [7, 1, 5, 3, 6, 4]
    # res = maxProfit(arr)
    # print('res = ', res)

    nums = [-2,1,-3,4,-1,2,1,-5,4]
    max_v = maxSubArray(nums)
    print(max_v)