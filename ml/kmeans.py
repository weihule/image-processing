import os
import numpy as np
import matplotlib.pyplot as plt
# import sklearn
# from sklearn.cluster import k_means
import pandas as pd
# from sklearn.cluster import k_means

def getData(path):
    data = []
    df = pd.read_csv(path)
    print(df.shape)
    print(df["Sepal.Length"][:5])
    # print(df.loc[1].values)

    for index in df.index:
        data.append(df.loc[index].values[1:-1])

    data = np.array(data)
    # print(data.shape)
    return data

def getData_dat(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[7:]:
            line = line.strip("\n")
            data.append(line.split(",")[:-1])

    data = np.array(data)
    return data

class Kmeans:
    def __init__(self, data, num_clusters):
        self.data = data
        self.num_clusters = num_clusters

    def train(self, max_iterations):
        # 1. 先随机选择k个中心点
        centroids = Kmeans.centroids_init(self.data, self.num_clusters)
        # 2. 开始训练
        num_examples = self.data.shape[0]
        closest_centroids_ids = np.empty((num_examples, 1))
        for _ in range(max_iterations):
            # 3. 得到当前每一个样本点到K个中心点的距离，找到最近的
            closest_centroids_ids = Kmeans.centroids_find_closest(self.data, centroids)
            # 4. 进行中心点距离更新
            centroids = Kmeans.centroids_compute(self.data, closest_centroids_ids, self.num_clusters)
        return centroids, closest_centroids_ids

    @staticmethod
    def centroids_find_closest(data, centroids):
        num_examples = data.shape[0]
        num_centroids = centroids.shape[0]
        closest_centroids_ids = np.zeros((num_examples, 1))
        # 遍历每一个样本数据点
        for example_index in range(num_examples):
            distance = np.zeros((num_centroids, 1))
            # 计算每个数据点到簇中心的距离
            for centroid_index in range(num_centroids):
                distance_diff = data[example_index, :] - centroids[centroid_index, :]
                distance[centroid_index] = np.sum(distance_diff**2)
            closest_centroids_ids[example_index] = np.argmin(distance)     # 返回最小值所在的下标
        return closest_centroids_ids

    @staticmethod
    def centroids_compute(data, closest_centroids_ids, num_clusters):
        num_features = data.shape[1]
        centroids = np.zeros((num_clusters, num_features))
        for centroid_id in range(num_clusters):
            # 找到当前属于这个簇中心点的数据
            closest_ids = closest_centroids_ids == centroid_id
            centroids[centroid_id] = np.mean(data[closest_ids.flatten(), :], axis=0)
        return centroids

    @staticmethod
    def centroids_init(data, num_clusters):
        num_examples = data.shape[0]
        random_ids = np.random.permutation(num_examples)    # random_ids 里面存放的是数据的下标
        centroids = data[random_ids[:num_clusters], :]

        return centroids

if __name__ == "__main__":
    # # data = getData_dat("./dataset/banana.dat")
    # data = getData("./dataset/iris.csv")
    # # plt.scatter(data[:, 0], data[:, 1], marker=".", label="see")
    # # plt.xlabel("SepalLengthCm")
    # # plt.ylabel("SepalWidthCm")
    # # plt.legend(loc="best")
    # # # plt.xticks([])
    # # # plt.yticks([])
    # # # plt.axis("off")
    # # plt.show()

    data = pd.read_csv("./dataset/iris.csv")

    # 按类别进行了可视化
    iris_types = ["setosa", "versicolor", "virginica"]

    x_axis = "Petal.Length"
    y_axis = "Petal.Width"

    # print(data[x_axis].values)
    # print(type(data[x_axis].values))

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # for iris_type in iris_types:
    #     plt.scatter(data[x_axis][data["Species"] == iris_type], data[y_axis][data["Species"] == iris_type], label=iris_type, marker=".")
    # plt.title("label known")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.scatter(data[x_axis][:], data[y_axis][:], marker=".")
    # plt.title("label unknown")
    # plt.show()

    num_examples = data.shape[0]
    x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))

    # 指定好训练所需的参数
    num_clusters = 3
    max_iterations = 50

    k_means = Kmeans(x_train, num_clusters)
    centroids, closest_centroids_ids = k_means.train(max_iterations)

    # 聚类效果展示
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for iris_type in iris_types:
        plt.scatter(data[x_axis][data["Species"] == iris_type], data[y_axis][data["Species"] == iris_type], label=iris_type, marker=".")
    plt.title("label known")
    plt.legend()

    plt.subplot(1, 2, 2)
    for centroid_id, centroid in enumerate(centroids):
        current_examples_index = (closest_centroids_ids == centroid_id).flatten()
        plt.scatter(data[x_axis][current_examples_index], data[y_axis][current_examples_index], marker=".", label=centroid_id)

    for centroid_id, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], c='black', marker="x")

    plt.legend()
    plt.title("label kmeans")
    plt.show()