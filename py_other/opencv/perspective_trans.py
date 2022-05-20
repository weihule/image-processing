import os
import random
import numpy as np
import json
import cv2
from PIL import Image
import matplotlib.pylab as plt

# ------------------------------------------------------------- #
# 绘图函数
def show_img():
    img_path = "D:\\workspace\\data\\opencv\\0019.jpg"
    label_path = "D:\\workspace\\data\\opencv\\0019.json"

    # img_path = "./data/third_6365.jpg"
    # label_path = "./data/third_6365.json"

    img = cv2.imread(img_path)
    print('img.shape = ', img.shape)

    char_poses = [[0, 50], [50, 98], [98, 140], [140, 178], [178, 214], [214, 239]]

    with open(label_path, "r", encoding="utf-8") as f:
        info = json.load(f)
        bbox_list = info["annotation"]["objects"]

    for i in bbox_list:
            if i['seal'] == '1':
                continue
            if i['name'] == 'Main body(top)':
                continue
            bbox = i['points']
            text = i['text']
        # print(text)
        # if text == '第一联 收据联 盖章有效遗失不补':
            bbox = np.array(bbox).reshape((-1, 2))
            img = cv2.polylines(img, [bbox], True, (0, 255, 0), 4)
            img_crop, perspectiv_m, dst = cropImage_rot(img, bbox[0], bbox[1], bbox[2], bbox[3])
            
            h, w, _ = img_crop.shape

            infer_dst = np.matrix(perspectiv_m, copy=False) * np.vstack((bbox.transpose(1, 0), np.ones(4)))

            dst_input = np.vstack((dst.transpose(1, 0), np.ones(4)))
            infer_src = np.matrix(perspectiv_m, copy=False).I * np.matrix(dst_input, copy=False)
            
            char_pose = get_char_pose(char_poses=char_poses,
                            dst = dst, 
                            perspective_m=perspectiv_m, 
                            crop_shape=[h, w],
                            resized_shape=(32, 240, 3))
            # print(perspectiv_m)
            # print('src = \n', bbox)
            # print('dst = \n', dst)
            # print('infer_dst = \n', np.int32(infer_dst))
            # print('infer_src= \n', np.int32(infer_src))

            # points = bbox.reshape(1, -1, 2).astype(np.float32)  ， 一个都不能少    
            # new_points = cv2.perspectiveTransform(points, perspectiv_m)

            # print(dst)
            points = dst.reshape(1, -1, 2).astype(np.float32) #二维变三维， 整形转float型
            new_points = cv2.perspectiveTransform(points, np.matrix(perspectiv_m).I)
            # print('new_points = \n', np.int32(new_points))

            # print('char_pose = ', char_pose)
            for bbox in char_pose:
                bbox = np.array(bbox, np.int32)
                img = cv2.polylines(img, [bbox], True, (255, 0, 0), 4)

            # src = np.array(dst, np.int32)
            # img = cv2.polylines(img, [src], True, (0, 0, 255), 4)
            
            # cv2.imshow('img_crop', img_crop)
            # cv2.waitKey(0)
            break

    cv2.namedWindow("res", cv2.WINDOW_FREERATIO)
    cv2.imshow("res", img)
    cv2.waitKey(0)


        # ===========
        # char_pose = char_pose * np.array(w / 240)
        # char_pose = np.tile(char_pose, 2).flatten()
        # char_pose[2], char_pose[3] = char_pose[3], char_pose[2]
        # pose = np.vstack(( char_pose, dst[:, 1], np.ones(shape=(4))) )
        # pose_src = np.matrix(perspectiv_m).I * np.matrix(pose)
        # pose_src = np.array(pose_src[:-1].transpose(1, 0), dtype=np.int32)
        # img = cv2.polylines(img, [pose_src], True, (255, 0, 0), 4)
        # ===========
        # print(type(char_pose))
        # char_pose = char_pose * np.array(w / 240)  
        # print(type(char_pose))
        # char_pose = np.hstack((char_pose, char_pose[:, 1].reshape((-1, 1)), \
        #                     char_pose[:, 0].reshape((-1, 1)) )).reshape((-1, 1, 4))
        # # print(char_pose)
        # char_pose = np.insert(char_pose, 1, dst[:, 1], axis=1)
        # char_pose = np.insert(char_pose, 2, np.ones(shape=(4)), axis=1) # (-1, 3, 4)
        # # print(char_pose, char_pose.shape)

        # pose_src = np.array([(np.matrix(perspectiv_m, copy=False).I * np.matrix(pose, copy=False))[:-1].transpose(1, 0) for pose in char_pose], dtype=np.int32)
        # # pose_src = np.array([pose[:-1].transpose(1, 0) for pose in pose_src])
        # # print(pose_src)



def get_char_pose(char_poses, dst, perspective_m, crop_shape, resized_shape):
    """
    得到切片中每个字的坐标
    input:
        char_poses: 输出的单字相对位置
        dst: 原始切片做透视变换的目标矩阵坐标
        perspective_m: 透视变换矩阵
        crop_shape: 切片的原始形状
        resized_shape: resize之后的形状
    output:
        返回一个三维数组 (B, 4, 2), B表示有多少个字, (4, 2)表示单个字的四个点坐标，以左上角为
        起始点，顺时针旋转
    """
    print('-'*20)
    # print(dst)
    crop_h, crop_w = crop_shape
    print(crop_h, crop_w)
    resized_h, resized_w, _ = resized_shape
    
    crop_info = crop_w if crop_h < crop_w * 3 else crop_h
    
    char_poses = np.array(char_poses) * np.float32(crop_info / resized_w)
    print(char_poses)
    if crop_h >= crop_w * 3:
        print('竖行')
        char_poses = np.repeat(char_poses, repeats=2, axis=1).reshape((-1, 1, 4))
        char_poses = np.insert(char_poses, 0, dst[:, 0], axis=1) # (-1, 2, 4)
    else:
        char_poses = np.hstack((char_poses, char_poses[:, 1].reshape((-1, 1)), char_poses[:, 0].reshape((-1, 1)))).reshape((-1, 1, 4))
        char_poses = np.insert(char_poses, 1, dst[:, 1], axis=1) # (-1, 2, 4)
    
    char_poses = np.transpose(char_poses, (0, 2, 1))     # (-1, 4, 2)

    inverse_pers_m = np.matrix(perspective_m, copy=False).I     # 透视变换矩阵的逆
    pose_src = np.int32([cv2.perspectiveTransform(np.float32(pose).reshape(1, -1, 2), inverse_pers_m) for pose in char_poses])
    print('-'*20)
    return pose_src.tolist()


    def get_char_pose(char_poses, dst, perspective_m, crop_shape, resized_shape):
        """
        得到切片中每个字的坐标
        input:
            char_poses: 输出的单字相对位置
            dst: 原始切片做透视变换的目标矩阵坐标
            perspective_m: 透视变换矩阵
            crop_shape: 切片的原始形状,包括 h, w
            resized_shape: resize之后的形状, 包括 h, w, c
        output:
            返回一个三维数组 (B, 4, 2), B表示有多少个字, (4, 2)表示单个字的四个点坐标，以左上角为
            起始点，顺时针旋转
        """
        crop_h, crop_w = crop_shape
        resized_h, resized_w, _ = resized_shape

        char_poses = char_poses * np.float32(crop_w / resized_w)
        char_poses = np.hstack((char_poses, char_poses[:, 1].reshape((-1, 1)), char_poses[:, 0].reshape((-1, 1)))).reshape((-1, 1, 4))

        char_poses = np.insert(char_poses, 1, dst[:, 1], axis=1)  # char_pose -> (-1, 2, 4)
        char_poses = np.insert(char_poses, 2, np.ones(4), axis=1)  # char_pose -> (-1, 3, 4)
        # print(char_poses, char_poses.shape)

        pose_src = np.array([(np.matrix(perspective_m, copy=False).I * np.matrix(pose, copy=False))[:-1].transpose(1, 0) for pose in char_poses], dtype=np.int32)

        return pose_src.tolist()


def cropImage_rot(img,pt1,pt2,pt3,pt4):
    pts = [pt1[0],pt1[1],pt2[0],pt2[1],pt3[0],pt3[1],pt4[0],pt4[1]]

    s, h, w = img.shape

    width = int((np.linalg.norm([pts[2]-pts[0],pts[3]-pts[1]]) + np.linalg.norm([pts[4]-pts[6],pts[5]-pts[7]])) / 2)
    height = int((np.linalg.norm([pts[6]-pts[0],pts[7]-pts[1]]) + np.linalg.norm([pts[4]-pts[2],pts[5]-pts[3]])) / 2)

    src = np.float32([[pts[0], pts[1]], [pts[2], pts[3]], [pts[4], pts[5]], [pts[6], pts[7]]])
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    src_add = np.vstack((src.transpose(1, 0), np.ones(4)))

    transform = cv2.getPerspectiveTransform(src, dst)
    transform = warp_perspective_matrix(src, dst)
    # print(transform)
    # print(src)
    # print(dst)
    img1 = cv2.warpPerspective(img, M=transform, dsize=(width, height))
    # print(np.array(np.matrix(transform)*np.matrix(src_add), np.int32))
    # plt.imshow(img1[..., ::-1])
    # plt.xticks([])
    # plt.yticks([])
    # plt.axis('off')
    # plt.show()

    return img1, transform, dst

def img_crop():
    img_path = "D:\\workspace\\proce_data\\third_6365.jpg"
    label_path = "D:\\workspace\\proce_data\\third_6365.json"

    img = Image.open(img_path)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

    with open(label_path, "r", encoding="utf-8") as json_file:
        infos_ = json.load(json_file)
        infos = infos_["annotation"]["objects"]
        count_per = 0
        for info in infos:

            count_per += 1
            text = info["text"]
            
            bbox = info["points"]
            # print(np.array(bbox))
            img_crop = cropImage_rot(img, bbox[0], bbox[1], bbox[2], bbox[3])
            # cv2.imshow("test", img_crop)
            # cv2.waitKey(0)

        #     postfix = ".png"
        #     crop_name = fn[:-4] + "_" + str(count_per).rjust(3, "0") + postfix
        #     save_path = os.path.join("./slices", crop_name)
        #     cv2.imencode(postfix, img_crop)[1].tofile(save_path)

        # img_crop = cropImage_rot(img, [2729,735], [2721,793], [3023,809], [3031,756])
        # print(img_crop.shape)
        # cv2.imencode(".png", img_crop)[1].tofile("./test.png")

def warp_perspective_matrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i, :] = [ A_i[0], A_i[1], 1, 
                             0,      0, 0,
                       -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1, :]   = [      0,      0, 0,
                        A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
 
    A = np.mat(A)
    warpMatrix = A.I * B #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    
    #之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

def median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half])/2

def show_center():
    char_center = [4.0, 9.0, 14.0, 20.0, 26.0, 31.0, 37.0, 42.0, 47.5, 53.0, 58.0]
    img = cv2.imread('test.png')
    print(img.shape)
    resize_img = cv2.resize(img, (240, 32))
    print(resize_img.shape)

    char_center = [p*4 for p in char_center]
    # 要画的点坐标
    points_list = [(int(p), 10) for p in char_center]
    for point in points_list:
        print(point)
        cv2.circle(resize_img, point, 2, (0, 0, 225), 2)
    cv2.namedWindow('test.png', cv2.WINDOW_NORMAL)
    cv2.imshow('test.png', resize_img)
    cv2.waitKey(0)


def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i, :] = [ A_i[0], A_i[1], 1, 
                             0,      0, 0,
                       -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1, :]   = [      0,      0, 0,
                        A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
 
    A = np.mat(A)
    warpMatrix = A.I * B #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    # print('A = \n', A)
    
    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) # 插入a_33 = 1
    # print('warpMatrix = \n', warpMatrix)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

def test():
    src_1 = [[395.0, 291.0], [624.0, 291.0], [1009.0, 457.0], [10.0, 457.0]]
    # src_1 = [[395.0, 291.0], [624.0, 291.0], [624.0, 457.0], [395.0, 457.0]]    # 正矩形
    src_1 = np.float32(src_1)
    dst_1 = [[46.0, 100.0], [600.0, 100.0], [600.0, 920.0], [46.0, 920.0]]
    dst_1 = np.float32(dst_1)
    

    src_2 = [[50.0, 0.0], [150.0, 0.0], [0.0, 200.0], [200.0, 200.0]]
    src_2 = np.float32(src_2)
    dst_2 = [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]]
    dst_2 = np.float32(dst_2)
    
    src = src_1
    dst = dst_1
    print('src = \n', src)
    print('dst = \n', dst)
    # warpMatrix = WarpPerspectiveMatrix(src, dst)
    warpMatrix = cv2.getPerspectiveTransform(src, dst)      # 3*3
    m = cv2.perspectiveTransform()
    print(warpMatrix)
    
    src_input = np.vstack((src.transpose(1, 0), np.ones(4)))
    print('src_input = \n', np.int32(src_input))
    infer_dst = np.matrix(warpMatrix, copy=False) * src_input

    print('infer_dst = \n', np.int32(infer_dst))

    img = np.zeros((1500, 1500, 3), np.uint8)
    img.fill(200)
    # 要画的点坐标
    for point in src:
        cv2.circle(img, np.int32(point), 2, (0, 0, 225), 4)
        cv2.putText(img, str(point), np.int32(point), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
    cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('img', img)
    cv2.waitKey(0)

if __name__ == "__main__":
    show_img()
    # show_center()

    # img_crop()

    # arr = np.array([[0.0, 0.0], [50.0, 0.0], [50.0, 60.0], [0.0, 60.0]])
    # print(arr.shape)

    # arr = np.array([[0, 290], [290, 530]])
    # arr = np.repeat(arr, 2, 1)
    # print(arr)
    





