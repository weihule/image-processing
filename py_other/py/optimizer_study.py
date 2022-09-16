import numpy as np

# 带动量的随机梯度下降SGD
def momentum_sgd():
    # 训练集，每个样本有三个分量
    x = np.array([(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (-1, -2), (-2, -1)])
    y = np.array([11, 20, 13, 22, 15, 24, -20, -13])

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    alpha = 0.01  # 学习率
    momentum = 0.1  # 动量
    threshold = 0.0001  # 停止迭代的错误阈值
    iterations = 1500  # 迭代次数
    error = 0  # 初始错误为0
    gradient = 0  # 初始梯度为0

    # 迭代开始
    for i in range(iterations):
        j = i % m
        print(x.shape, theta.shape, np.dot(x, theta))
        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
                                     (np.dot(x, theta) - y))
        # 迭代停止
        if abs(error) <= threshold:
            break

        gradient = momentum * gradient + alpha * (x[j] *
                                                  (np.dot(x[j], theta) - y[j]))
        theta -= gradient
        if i == 1:
            break

    print('迭代次数：%d' % (i + 1), 'theta: ', theta, 'error: %f' % error)


if __name__ == '__main__':
    momentum_sgd()

