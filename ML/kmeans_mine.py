from cProfile import label
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters
        self.num_examples = data.shape[0]
    
    def train(self, max_iterations):
        # 1. 先随机选择k个中心点, (num_clusters, data.shape[1])
        centroids = Kmeans.centroids_init(self.data, self.num_clusters)
        # 2. 开始训练
        # 存放的是每个数据点属于哪个簇的簇下标
        closest_centroids_ids = np.empty((self.num_examples, 1))    # (self.num_examples, 1)   
        for _ in range(max_iterations):
            # 3. 得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centroids_ids = Kmeans.centroids_find_closest(self.data, centroids)
            # print(closest_centroids_ids[:7])
            # 4. 进行中心点更新
            centroids = Kmeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)

        print(centroids)
        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_init(data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clusters]]

        return centroids

    @staticmethod
    def centroids_find_closest(data, centroids):
        num_centroids = centroids.shape[0]
        num_examples = data.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        # 遍历每一个样本数据点
        for example_index in range(num_examples):
            dis = np.zeros((num_centroids, 1))
            # 计算每个样本数据点到簇中心的距离
            for centroid_index in range(num_centroids):
                dis_diff = data[example_index] - centroids[centroid_index]
                dis[centroid_index][0] = np.sum(dis_diff**2)
            closest_centroids_ids[example_index][0] = np.argmin(dis)
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        num_examples = data.shape[0]
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters, num_features))
        for centroid_ids in range(num_clusters):
            # 找到当前属于这个簇中心点的数据
            closest_ids = closest_centroids_ids == centroid_ids
            centroids[centroid_ids] = np.mean(data[closest_ids.flatten()], axis=0)
        return centroids

if __name__ == "__main__":
    data = pd.read_csv("./dataset/iris.csv")
    # print(type(data))
    # print(data["Petal.Length"][data["Species"] == "setosa"])

    iris_types = ["setosa", "versicolor", "virginica"]

    x_axis = "Petal.Length"
    y_axis = "Petal.Width"

    num_examples = data.shape[0]
    x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
    # print(x_train.shape)
    # 指定训练所需的参数
    num_clusters = 3
    max_iterations = 50

    k_means = Kmeans(x_train, num_clusters)
    centroids, closest_centroids_ids = k_means.train(max_iterations)

    # 进行数据的可视化
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    for type in iris_types:
        plt.scatter(data[x_axis][data["Species"] == type], data[y_axis][data["Species"] == type], marker=".", label=type)
    plt.title("label known")
    plt.legend()

    plt.subplot(1, 2, 2)
    for index in range(num_clusters):
        current_examples_index = (closest_centroids_ids == index).flatten()
        plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], marker=".", label=index)
    plt.title("label kmeans")
    plt.legend()

    plt.show()



