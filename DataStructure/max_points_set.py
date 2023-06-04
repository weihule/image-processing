def findpoints():
    nums = 5
    points = [[1, 2], [5, 3], [4, 6], [7, 5], [9, 0]]
    result = []
    print(points)
    points.sort()
    print(points)
    # 先确定最右边的点p，其余的边界点一定是y>py，x<px
    max_x2y = points[- 1][1]
    result.append(points[- 1])
    # 倒着往回找点
    for j in range(nums - 2, -1, -1):
        if points[j][1] > max_x2y:
            result.append(points[j])
            max_x2y = points[j][1]
    while len(result) > 0:
        point = result.pop()
        print(point[0], point[1])


if __name__ == "__main__":
    findpoints()