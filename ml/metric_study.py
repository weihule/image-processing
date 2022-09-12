from sklearn.metrics  import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import torch
                                                                                                                                                                                                                                                                                  

def offical_func(scores, labels):
    precision, recall, thres = precision_recall_curve(labels, scores)
    print(precision, recall, thres)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()



if __name__ == "__main__":
    # score = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
    # label = np.array([1, 1, 1, 1, 0, 0, 0])
    # offical_func(score, label)

    seed = 0
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子

    
    dist = torch.randn(4, 4)
    target = torch.tensor([3, 0, 1, 2]).long()
    mask = torch.eq(target.expand(4, 4), target.expand(4, 4).T)
    for i in range(4):
        temp = dist[i][mask[i]].max()
        print(temp)

