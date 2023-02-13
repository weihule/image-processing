import os
import cv2
import numpy as np
import PIL


def crop_image(img, location):
    crop_img = img[location[1]: location[3], location[0]: location[2]]
    return crop_img


def main():
    # 0113_c5s1_018801_00.jpg  行人遮挡  
    # 0813_c6s2_092218_00.jpg  光照变化
    # 1277_c2s3_020082_00.jpg  行人遮挡
    # 1125_c6s3_017892_00.jpg  视角变化
    img_name = "0113_c5s1_018801_00.jpg"
    path1 = "D:\\Desktop\\tempfile\\some_infer\\reid_infer\\market1501\\osnet\\no_re_rank\\" + img_name
    path2 = "D:\\Desktop\\tempfile\\some_infer\\reid_infer\\market1501\\osnet\\re_rank\\" + img_name
    img1 = cv2.imread(path1)
    img1 = crop_image(img1, (100, 50, 1500, 320))
    img2 = cv2.imread(path2)
    img2 = crop_image(img2, (100, 50, 1500, 320))

    combine_img = np.concatenate((img1, img2), axis=0)
    fill = np.ones((combine_img.shape[0], 90, 3), dtype=np.uint8) * 255
    combine_img = np.concatenate((fill, combine_img), axis=1)
    cv2.putText(combine_img, "baseline", np.int32([20, 150]), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
    cv2.putText(combine_img, "after", np.int32([40, 420]), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
    cv2.imshow("res", combine_img)
    cv2.waitKey(0)
    save_path = r"D:\Desktop\tempfile\reid_paper\some_pics\行人遮挡2.jpg"
    cv2.imencode(".png", combine_img)[1].tofile(save_path)


def test():
    dic = {"1": 1, "2": 2}
    for k, v in dic.items():
        print(k , v)
    print(type(dic))

    arr = [1, 2]
    print(type(arr))

    strs = "{'public': {'1': 0}, 'workfile': r'D:\\Desktop\\tempfile\\个人电子签名.png'}"
    print(strs)
    eval_strs = eval(strs)
    for k, v in eval_strs.items():
        if k == "workfile":
            print(os.path.exists(v), v)
    
    arr = "SAP’D\\Rc"
    print(arr.split("’D：\\"))

if __name__ == "__main__":
    # main()

    test()