import os
import shutil
import torch
import numpy as np
from torch.nn.functional import softmax

print(torch.__version__)

# outputs = torch.tensor([[0.85, 0.4, 0.1, 0.85, 0.4], [0.9, 0.2, 0.3, 0.8, 0.7],
#                         [0.2, 0.1, 0.9, 0.5, 0.3],[0.2, 0.2, 0.4, 0.9, 0.6],
#                         [0.9, 0.5, 0.1, 0.5, 0.4], [0.9, 0.1, 0.4, 0.1, 0.6]])

# outputs = torch.randn((6, 5))
# print(outputs)

# _, predicted = torch.max(outputs, dim=-1)

# print(predicted)
# print(predicted.shape)

# print(torch.argmax(outputs,dim=-1))

# correct = (predicted == label)

# print("predicted:", predicted)
# print("label:", label)
# print(correct)
# print(correct[1].item())

# pre2 = torch.argmax(outputs, dim=1)
# print("pre2:", pre2)	# 输出结果和predicted一样

# root = "./dataset"

# train_img_path = []
# train_img_label = []
# for fn in glob.glob(os.path.join(root, "train/*/*")):
#     print(fn)
#     print(fn.split(os.sep)[2])

# a = np.array([1, 2, 3])
# b = torch.tensor(a)
# c = torch.as_tensor(a)

# print(a, id(a))
# print(b, id(b))
# print(c, id(c))

# img = Image.open("../pred_pic/daisy01.jpg")
# transform1 = transforms.RandomHorizontalFlip()
# transform2 = transforms.RandomVerticalFlip()

# img1 = transform1(img)
# img2 = transform1(img)

# # fig = plt.figure()
# # ax1 = fig.add_subplot(1, 2, 1)
# # ax2 = fig.add_subplot(1, 2, 2)
# fig,ax = plt.subplots(nrows=1, ncols=3)
# axes = ax.flatten()
# axes[0].imshow(img)
# axes[0].set_title("src")
# axes[0].get_yaxis().set_visible(False)
# axes[0].get_xaxis().set_visible(True)
# axes[1].imshow(img1)
# axes[2].imshow(img2)
# # plt.imshow(img)
# plt.axis("off")
# # plt.xticks([])  #去掉横坐标值
# # plt.yticks([])  #去掉纵坐标值
# plt.show()


a = [1, 2, 3]
a = np.array(a).reshape((1, 3))
tensor_a = torch.tensor(a, dtype=torch.float32)
res = softmax(tensor_a , dim=1)
print(res)

num = np.sum(np.power(np.e, [1, 2, 3]))
result = np.power(np.e, 2) / num
print(result)