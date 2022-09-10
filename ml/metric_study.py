from sklearn.metrics  import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
                                                                                                                                                                                                                                                                                  

def offical_func(scores, labels):
    precision, recall, thres = precision_recall_curve(labels, scores)
    print(precision, recall, thres)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()

if __name__ == "__main__":
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
    label = np.array([1, 1, 1, 1, 0, 0, 0])
    offical_func(score, label)


