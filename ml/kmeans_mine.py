import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self, datas, num_clusters):
        self.datas = datas
        self.num_clusters = num_clusters
        self.num_examples = datas.shape[0]
        self.num_features = datas.shape[1]
    
    def train(self, max_iterations):
        # 1. 先随机选择k个中心点, centroids shape is (num_clusters, datas.shape[1])
        centroids = self.centroids_init(self.datas, self.num_clusters)
        # 2. 开始训练
        # 存放的是每个数据点属于哪个簇的簇下标
        closest_centroids_ids = np.empty((self.num_examples, 1))    # (self.num_examples, 1)   
        for _ in range(max_iterations):
            # 3. 得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centroids_ids = self.centroids_find_closest(self.datas, centroids)

            # 4. 进行中心点更新
            centroids = self.centroids_compute(self.datas, closest_centroids_ids, self.num_clusters)

        # print(centroids)
        return centroids, closest_centroids_ids

    def centroids_init(self, datas, num_clusters):
        """
        从所有样本中随机挑选三组作为初始化中心点
        """
        # np.random.permutation 随机排序序列, 这里相当于将[0, 1, ..., 149] 打乱顺序随机排列
        random_ids = np.random.permutation(self.num_examples)   # shape is [150]
        centroids = datas[random_ids[:num_clusters]]        # shape is [self.num_examples, 2]

        return centroids

    def centroids_find_closest(self, datas, centroids):
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((self.num_examples, 1))
        # 遍历每一个样本数据点
        for example_index in range(self.num_examples):
            dis = np.zeros((num_centroids, 1))
            # 计算每个样本数据点到K个簇中心的距离
            for centroid_index in range(num_centroids):
                dis_diff = datas[example_index] - centroids[centroid_index]
                dis[centroid_index][0] = np.sum(dis_diff**2)
            closest_centroids_ids[example_index][0] = np.argmin(dis)
        return closest_centroids_ids

    def centroids_compute(self, datas, closest_centroids_ids, num_clusters):
        centroids = np.zeros((num_clusters, self.num_features))
        for centroid_ids in range(num_clusters):
            # 找到当前属于这个簇中心点的数据
            closest_ids = closest_centroids_ids == centroid_ids
            centroids[centroid_ids] = np.mean(datas[closest_ids.flatten()], axis=0)
        return centroids


def show_res(datas, class_types, x_axis, y_axis, trained_datas):
    """
    可视化
    """
    # 进行数据的可视化
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    for type in class_types:
        mask = datas["Species"] == type
        plt.scatter(datas[x_axis][mask], datas[y_axis][mask], marker=".", label=type)
    plt.title("label known")
    plt.legend()

    centroids = trained_datas[0]
    closest_centroids_ids = trained_datas[1]
    plt.subplot(1, 2, 2)
    for index in range(num_clusters):
        current_examples_index = (closest_centroids_ids == index).flatten()
        plt.scatter(datas[x_axis][current_examples_index], datas[y_axis][current_examples_index], marker=".", label=index)
    plt.title("label kmeans")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    datas = pd.read_csv("D:\\workspace\\data\\ml\\iris.csv")    # shape is [150, 6]

    iris_types = ["setosa", "versicolor", "virginica"]

    x_axis = "Petal.Length"
    y_axis = "Petal.Width"
    num_examples = datas.shape[0]
    x_train = datas[[x_axis, y_axis]].values.reshape((num_examples, 2))
    # 指定训练所需的参数
    num_clusters = 3
    max_iterations = 50

    k_means = Kmeans(x_train, num_clusters)
    centroids, closest_centroids_ids = k_means.train(max_iterations)

    # show_res(datas, iris_types, x_axis, y_axis, [centroids, closest_centroids_ids])


