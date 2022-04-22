import os
import numpy as np
import json
import cv2
from PIL import Image

# ------------------------------------------------------------- #
# 绘图函数
def show_img():
    img_path = "./data/third_6370.jpg"
    label_path = "./data/third_6370.json"

    # img_path = "./data/third_6365.jpg"
    # label_path = "./data/third_6365.json"

    img = cv2.imread(img_path)
    pt1 = (2729,735)
    pt2 = (3023,809)
    # cv2.rectangle(img, (2729,735), (3023,809), (0, 255, 0), 2)

    # 要画的点坐标
    points_list = [(2729,735), (2721, 793), (3023, 809)]
    for point in points_list:
        cv2.circle(img, point, 2, (0, 0, 225), 8)

    # with open(label_path, "r", encoding="utf-8") as f:
    #     info = json.load(f)
    #     bbox_list = info["annotation"]["objects"]

    # for i in bbox_list:
    #     bbox = i["points"]
    #     bbox = np.array(bbox).reshape((-1, 1, 2))
    #     img = cv2.polylines(img, [bbox], True, (0, 255, 0), 4)

    cv2.namedWindow("res",cv2.WINDOW_KEEPRATIO)
    cv2.imshow("res", img)
    cv2.waitKey(0)


def cropImage_rot(img,pt1,pt2,pt3,pt4):
    pts = [pt1[0],pt1[1],pt2[0],pt2[1],pt3[0],pt3[1],pt4[0],pt4[1]]

    s, h, w = img.shape

    width = int((np.linalg.norm([pts[2]-pts[0],pts[3]-pts[1]]) + np.linalg.norm([pts[4]-pts[6],pts[5]-pts[7]])) / 2)
    height = int((np.linalg.norm([pts[6]-pts[0],pts[7]-pts[1]]) + np.linalg.norm([pts[4]-pts[2],pts[5]-pts[3]])) / 2)

    src = np.float32([[pts[0], pts[1]], [pts[2], pts[3]], [pts[4], pts[5]], [pts[6], pts[7]]])
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(src, dst)
    img1 = cv2.warpPerspective(img, M=transform, dsize=(width, height))

    return img1

def img_crop():
    fn = "third_6370.jpg"
    img_path = "./data/third_6370.jpg"
    label_path = "./data/third_6370.json"

    # fn = "third_6365.jpg"
    # img_path = "./data/third_6365.jpg"
    # label_path = "./data/third_6365.json"

    img = Image.open(img_path)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

    with open(label_path, "r", encoding="utf-8") as json_file:
        infos_ = json.load(json_file)
        infos = infos_["annotation"]["objects"]
        count_per = 0
        # for info in infos:
        #     count_per += 1
        #     text = info["text"]
        #     bbox = info["points"]
        #     img_crop = cropImage_rot(img, bbox[0], bbox[1], bbox[2], bbox[3])

        #     postfix = ".png"
        #     crop_name = fn[:-4] + "_" + str(count_per).rjust(3, "0") + postfix
        #     save_path = os.path.join("./slices", crop_name)
        #     cv2.imencode(postfix, img_crop)[1].tofile(save_path)

        img_crop = cropImage_rot(img, [2729,735], [2721,793], [3023,809], [3031,756])
        print(img_crop.shape)
        cv2.imencode(".png", img_crop)[1].tofile("./test.png")

if __name__ == "__main__":
    show_img()

    # img_crop()


