
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


if __name__ == "__main__":
    # n = 4
    # res = pa_lou(n)
    # print('res = ', res)

    arr = [7, 1, 5, 3, 6, 4]
    res = maxProfit(arr)
    print('res = ', res)