import os
import numpy as np
import torch

if __name__ == "__main__":
    # start = time.clock()
    # arr1 = [1, 1, 4, 1, 4, 6, 1, 6]
    # arr2 = [2, 3, 7, 3, 7, 9, 2, 9]
    # area1 = Polygon(np.array(arr1).reshape(4, 2))
    # area2 = Polygon(np.array(arr2).reshape(4, 2))
    # iou = area1.intersection(area2).area / (area1.area + area2.area)



    # ar1 = [1, 1, 4, 6]
    # ar1 = [10, 10, 14, 14]
    # ar2 = [2, 3, 7, 9]
    # iou = get_iou(ar1, ar2)

    # print(f'{iou}, running time: {time.clock()-start}')


    # pred_bbox = torch.randn(7, 4)
    # print(pred_bbox, pred_bbox[:, 1])
    # a1 = (pred_bbox[:, 2] - pred_bbox[:, 0]) * (pred_bbox[:, 3] - pred_bbox[:, 1])
    # b1 = torch.tensor(10)
    # print(a1)
    # print(a1+b1)
    # temp = pred_bbox.new_zeros(10, 10)
    # print(len(temp), temp.numel())

    # features = np.array([[0, 0, 0, 0],
    #             [0, 0, 0, 1],
    #             [0, 1, 0, 1],
    #             [0, 1, 1, 0],
    #             [0, 0, 0, 0],
    #             [1, 0, 0, 0],
    #             [1, 0, 0, 1],
    #             [1, 1, 1, 1],
    #             [1, 0, 1, 2],
    #             [1, 0, 1, 2],
    #             [2, 0, 1, 2],
    #             [2, 0, 1, 1],
    #             [2, 1, 0, 1],
    #             [2, 1, 0, 2],
    #             [2, 0, 0, 0]])

    # labels = np.array(['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes']).reshape((-1, 1))

    # # dataset = np.hstack((features, labels))
    # dataset = np.concatenate((features, labels), axis=1)
    # mask = features[:, 2] == 0		# 根据索引为2的列生成mask
    # subdataset = features[mask]		# 将mask为false的行全都去除
    # print(subdataset)
    # subdataset = np.delete(subdataset, 2, axis=1)	# 删除索引为2的列


    # a = torch.tensor([2, 4, 6, 8])
    # b = torch.tensor([4., 8., 16., 32.])

    # a = torch.rand(1, 7)
    # print(a.max(dim=1))

    a = torch.rand(4, 3, 14, 14)
    print(a.shape[-2:])
