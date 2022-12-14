
def solution(p1, p2):
    """
    p1: 打井费用
    p2: 管道费用
    """
    min_price = min(p1)
    min_v = float('inf')
    idxes = [0, 0]
    for i in range(len(p1)):
        for j in range(len(p1)):
            if p2[i][j] < min_price and p2[i][j] < min_v:
                min_v = p2[i][j]
                idxes[0], idxes[1] = i, j
    visited = []
    result = 0
    if min_v != float('inf'):
        visited.append(idxes[0])
        visited.append(idxes[1])
        result += p2[idxes[0]][idxes[1]]
        
        for i in range(len(p1)):
            if i in visited:
                continue
            else:
                index, p1_flag = price_compare(i, visited, p1, p2)
                if p1_flag:
                    result += p1[i]
                    visited.append(i)
                else:
                    result += p2[i][index]
                    visited.append(index)
    else:
        visited.append(p1.index(min_price))
        result += min_price
        for i in range(n):
            if i in range(n):
                continue
            else:
                index, p1_flag = price_compare(i, visited, p1, p2)
                if p1_flag:
                    result += p1[i]
                    visited.append(i)
                else:
                    result += p2[i][index]
                    visited.append(index)
    return result
                    
                
def price_compare(i, visited, p1, p2):
    p1 = p1[i]
    min_idx = -1
    f = True
    min_cost = float("inf")
    for j in visited:
        if p2[i][j] < p1 and p2[i][j] < min_cost:
            min_cost = j
            f = False
            min_cost = p2[i][j]
    return min_idx, f

if __name__ == "__main__":
    n = 6

    # 打井费用
    prices = [5, 4, 4, 3, 1, 20]

    # 简历管道费用
    prices2 = [[0, 2, 2, 2, 9, 9],
               [2, 0, 3, 3, 9, 9],
               [2, 3, 0, 4, 9, 9],
               [2, 3, 4, 0, 9, 9],
               [9, 9, 9, 9, 0, 9],
               [9, 9, 9, 9, 9, 0]]

    final_cost = solution(prices, prices2)
    print(final_cost)
