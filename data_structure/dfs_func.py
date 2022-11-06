class Solution:
    def generateParenthesis(self, n: int):
        res = []
        cur_str = ''

        def dfs(cur_str, left, right):
            """
            :param cur_str: 从根结点到叶子结点的路径字符串
            :param left: 左括号还可以使用的个数
            :param right: 右括号还可以使用的个数
            :return:
            """
            if left == 0 and right == 0:
                res.append(cur_str)
                return
            if right < left:
                return
            if left > 0:
                dfs(cur_str + '(', left - 1, right)
            if right > 0:
                dfs(cur_str + ')', left, right - 1)

        dfs(cur_str, n, n)
        return res


def digui(max_v):
    if max_v <= 100 and max_v >= 0:
        res = max_v + digui(max_v - 1)
        return res
    else:
        return 0


def sum_func(max_v=100):
    res = 0
    for i in range(max_v):
        res += i
    return res


def binary_find(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (right - left) // 2
        if arr[mid] == target:
            return mid
        elif target > arr[mid]:
            left = mid + 1
        else:
            right = mid - 1
    # 返回-1表示未查找到
    return -1


if __name__ == "__main__":
    # sol = Solution()
    # res = sol.generateParenthesis(n=3)

    # print(res)

    # class Student(object): 
    #     name = 'Student' 
    #     def __init__(self, name): 
    #         self.name = name 

    # s = Student("sam") 
    # s.name="bob" 
    # del s.name 
    # print(s.name)

    # mylist=[0,1,2,3,4,5,6,7,8] 
    # print(mylist[-6:-3])


    import torch
    print(torch.cuda.is_available())



