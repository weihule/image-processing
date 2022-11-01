import operator
import os
import pickle
from math import log
import matplotlib.pyplot as plt
import numpy as np


def createDataset():
    dataset = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    features = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataset, features

def createTree(dataset, labels, feature_labels):
    '''
    dataset: 递归调用,每次传进来一个不同的数据集
    labels: 当前经过一次分支之后，是否分的干净
    feature_labels: 存储节点的顺序
    '''
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]  
     
    # 如果datset中只剩下一列标签数据
    if len(dataset[0]) == 1:
        return majority_cnt(classlist)
    
    labels_name = set(dataset[:, -1])
    best_feat_index = choose_best_feat(dataset, labels_name)
    
    best_feat_label = labels[best_feat_index]
    # print(best_feat_index, best_feat_label)
    feature_labels.append(best_feat_label)
     
    myTree = {best_feat_label:{}}
    del labels[best_feat_index]
    
    feat_values = dataset[:, best_feat_index]
    feat_values = set(feat_values)
    
    # 对当前节点下的每一个属性做分支
    for value in feat_values:
        sub_labels = labels
        sub_dataset = splitdataset(dataset, best_feat_index, value)
        myTree[best_feat_label][value] = createTree(sub_dataset, sub_labels, feature_labels)
        
    return myTree

def majority_cnt(classlist):
    '''计算当前节点当中,哪个类别最多的'''
    info = {}
    for i in classlist:
        if i not in info.keys():
            info[i] = 0
        info[i] += 1
        
    res = sorted(info.items(), key=operator.itemgetter(1))
    return res[-1][0]

def choose_best_feat(dataset, labels_name):
    num_features = len(dataset[0]) - 1
    base_entropy = calculate_entropy(dataset)   # 原始数据的熵
    best_info_gain = 0
    best_feature = -1
    # 开始遍历每一特征(每一列)
    for i in range(num_features):
        curr_feat = dataset[:, i]   # 当前列的所有属性
        curr_num_attr = len(curr_feat)  # 当前列的属性数量
        curr_attr_info = {}     # 统计当前列中各个属性的数量和对应的label
        for index, attr in enumerate(curr_feat):
            if attr not in curr_attr_info.keys():
                curr_attr_info[attr] = {}
                curr_attr_info[attr]['labels'] = {}
                for label in labels_name:
                    curr_attr_info[attr]['labels'][label] = 0
                curr_attr_info[attr]['num'] = 0
            curr_attr_info[attr]['labels'][dataset[index, -1]] += 1
            curr_attr_info[attr]['num'] += 1
        # print(curr_attr_info)
        
        # 开始计算当前特征的熵
        curr_entropy = 0
        for k, v in curr_attr_info.items():
            sub_entropy = 0
            for sub_k, sub_v in v['labels'].items():
                if sub_v == 0:
                    continue
                prob = float(sub_v / v['num'])
                sub_entropy -= prob * log(prob, 2)
            curr_entropy += float(v['num'] / curr_num_attr) * sub_entropy
            
        info_gain = base_entropy - curr_entropy
        # print(curr_entropy, info_gain)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
                
    return best_feature
                 
def splitdataset(dataset, best_feat_index, value):
    mask = dataset[:, best_feat_index] == value
    subdataset = dataset[mask]
    subdataset = np.delete(subdataset, best_feat_index, axis=1)
    return subdataset
        
def calculate_entropy(dataset):
    '''计算数据的熵'''
    dataset = np.array(dataset)
    num_examples = len(dataset)
    label_count = {}
    for label in dataset[:, -1]:
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] += 1
        
    entropy = 0
    for k in label_count.keys():
        prob = float(label_count[k] / num_examples)
        entropy -= prob * log(prob, 2)
        
    return entropy


if __name__ == "__main__":
    dataset, features = createDataset()
    dataset = np.array(dataset)
    labels_name = set(dataset[:, -1])
    # res = choose_best_feat(dataset, labels_name)
    feature_list = []
    tree = createTree(dataset, features, feature_list)
    print(tree)

        