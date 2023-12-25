from pathlib import Path
import numpy as np
import cv2


def main():
    root = "D:\Desktop\Mosaic"

    # 计算中心点
    center = (600, 600)

    # 创建一个1200x1200的黑色背景图
    background = np.zeros((1270, 1200, 3), dtype=np.uint8)

    # 读取四张图片
    image1 = cv2.imread(r'D:\Desktop\Mosaic\1638316731917383815.jpg')  
    image2 = cv2.imread(r'D:\Desktop\Mosaic\1683817904011797706.jpg')
    image3 = cv2.imread(r'D:\Desktop\Mosaic\-2412890800938252113.jpg')
    image4 = cv2.imread(r'D:\Desktop\Mosaic\-2676375563074623960.jpg')

    # 确保图片加载成功
    if image1 is None or image2 is None or image3 is None or image4 is None:
        print("Error: Unable to load one or more images.")
        exit()

    # 获取图片尺寸
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    h3, w3 = image3.shape[:2]
    h4, w4 = image4.shape[:2]

    # 计算中心点
    center = (600, 600)

    # 计算图片放置的位置
    x1, y1 = center[0] - w1, center[1] - h1
    x2, y2 = center[0], center[1] - h2
    x3, y3 = center[0] - w3, center[1]
    x4, y4 = center[0], center[1]

    # 将图片放置到背景图上
    background[y1:y1+h1, x1:x1+w1] = image1
    background[y2:y2+h2, x2:x2+w2] = image2
    background[y3:y3+h3, x3:x3+w3] = image3
    background[y4:y4+h4, x4:x4+w4] = image4

    # 保存结果图片
    cv2.imwrite('result_image.jpg', background)

    # 显示结果图片（可选）
    cv2.imshow('Result Image', background)
    cv2.waitKey(0)

    cv2.imwrite(r'D:\Desktop\result_image.jpg', background)



def mixup():
    # 读取两张图片
    image1 = cv2.imread(r'D:\Desktop\Mosaic\1638316731917383815.jpg')  # 替换为你的图片路径
    image2 = cv2.imread(r'D:\Desktop\Mosaic\1683817904011797706.jpg')  # 替换为你的图片路径

    # 确保图片加载成功
    if image1 is None or image2 is None:
        print("Error: Unable to load one or more images.")
        exit()

    # 将图像大小调整为相同尺寸，确保两张图像可以混合
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    min_h, min_w = min(h1, h2), min(w1, w2)

    image1 = cv2.resize(image1, (min_w, min_h))
    image2 = cv2.resize(image2, (min_w, min_h))

    # 指定 MixUp 操作的 alpha 值（0 到 1 之间）
    alpha = 0.5

    # 将图像按照指定的 alpha 值进行混合
    mixed_img = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)


    # 显示原始图像和混合后的图像（可选）
    cv2.imshow('Mixed Image', mixed_img)
    cv2.waitKey(0)

    cv2.imwrite(r'D:\Desktop\mixup.jpg', mixed_img)


if __name__ == "__main__":
    # main()

    mixup()
