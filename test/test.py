import os
import cv2


def main():
    img_path = r"D:\Desktop\001.png"
    img = cv2.imread(img_path)
    # width, height
    img = cv2.resize(img, (295, 413))
    cv2.imwrite(r"D:\Desktop\001.jpg", img)


if __name__ == "__main__":
    main()
