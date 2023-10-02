import os
import math
import cv2
import numpy as np
import random
import torch

import models
import decodes
from utils.util import compute_macs_and_params
from datasets.coco import COCO_CLASSES, COCO_CLASSES_COLOR
from datasets.voc import VOC_CLASSES, VOC_CLASSES_COLOR


def load_image(image, cfgs, divisor=32):
    assert cfgs["image_resize_style"] in ['retinastyle',
                                          'yolostyle'], 'wrong style!'

    h, w, _ = image.shape

    # normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255.)
    if cfgs["image_resize_style"] == "yolostyle":
        scale = cfgs["input_image_size"] / max(h, w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)

        image = cv2.resize(image, (resize_w, resize_h))

        pad_w = 0 if resize_w % divisor == 0 else divisor - resize_w % divisor
        pad_h = 0 if resize_h % divisor == 0 else divisor - resize_h % divisor

        padded_img = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                              dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = image
    else:
        padded_img = ""
        scale = ""

    return padded_img, scale


def inference(cfgs):
    assert cfgs["trained_dataset_name"] in ['COCO', 'VOC'], 'Unsupported dataset!'
    assert cfgs["model"] in models.__dict__.keys(), 'Unsupported model!'
    assert cfgs["decoder"] in decodes.__dict__.keys(), 'Unsupported decoder!'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if cfgs["seed"]:
        seed = cfgs["seed"]
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    model = models.__dict__[cfgs["model"]](
        **{"num_classes": cfgs["trained_num_classes"]}
    )
    decoder = decodes.__dict__[cfgs["decoder"]](
        **{
            "nms_type": 'python_nms',
            "nms_threshold": 0.5
        }
    )
    if cfgs["trained_model_path"]:
        saved_model = torch.load(cfgs["trained_model_path"],
                                 map_location=torch.device('cpu'))
        model.load_state_dict(saved_model)

    model = model.cuda()
    # decoder = decoder.cuda()

    model.eval()

    macs, params = compute_macs_and_params(cfgs["input_image_size"], model, device)
    print(f'model: {cfgs["model"]}, macs: {macs}, params: {params}')

    cap = cv2.VideoCapture(cfgs["video_path"])

    if not cap.isOpened():
        raise IOError("Couldn't open video!")

    # 设置视频编码器（FourCC）、帧率（FPS）和视频大小的参数，
    # 以便将视频帧写入到一个视频文件中。
    video_FourCC = cv2.VideoWriter_fourcc(*"XVID")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    if cfgs["save_video_path"]:
        print(f"save_video_info: {video_FourCC}, {video_fps}, {video_size}")
        out_video = cv2.VideoWriter(
            os.path.join(cfgs["save_video_path"], 'detection_result.avi'),
            video_FourCC, video_fps, video_size)

    if cfgs["trained_dataset_name"] == 'COCO':
        dataset_classes_name = COCO_CLASSES
        dataset_classes_color = COCO_CLASSES_COLOR
    else:
        dataset_classes_name = VOC_CLASSES
        dataset_classes_color = VOC_CLASSES_COLOR

    frame_num = 0
    while True:
        # cap.read()函数返回的第1个参数ret是一个布尔值, 表示当前这一帧是否获取正确
        ret, frame = cap.read()

        if not ret:
            break
        origin_img = frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img, scale = load_image(image, cfgs)
        resized_img = torch.tensor(resized_img)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
        # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
        outputs = model(resized_img.permute(2, 0, 1).float().unsqueeze(0).cuda())

        # batch_scores shape:[batch_size,max_object_num]
        # batch_classes shape:[batch_size,max_object_num]
        # batch_bboxes shape[batch_size,max_object_num,4]
        scores, classes, boxes = decoder(outputs)

        boxes /= scale

        # 去掉batch维度
        scores = scores.squeeze(0)
        classes = classes.squeeze(0)
        boxes = boxes.squeeze(0)

        scores = scores[classes > -1]
        boxes = boxes[classes > -1]
        classes = classes[classes > -1]

        boxes = boxes[scores > cfgs["min_score_threshold"]]
        classes = classes[scores > cfgs["min_score_threshold"]]
        scores = scores[scores > cfgs["min_score_threshold"]]

        # clip boxes
        origin_h, origin_w = origin_img.shape[0], origin_img.shape[1]
        boxes[:, 0] = np.maximum(boxes[:, 0], 0)
        boxes[:, 1] = np.maximum(boxes[:, 1], 0)
        boxes[:, 2] = np.minimum(boxes[:, 2], origin_w)
        boxes[:, 3] = np.minimum(boxes[:, 3], origin_h)

        # draw all pred boxes
        for per_score, per_class_index, per_box in zip(scores, classes, boxes):
            per_score = per_score.astype(np.float32)
            per_class_index = per_class_index.astype(np.int32)
            per_box = per_box.astype(np.int32)

            class_name, class_color = dataset_classes_name[per_class_index], dataset_classes_color[per_class_index]

            left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2], per_box[3])
            cv2.rectangle(origin_img,
                          left_top,
                          right_bottom,
                          color=class_color,
                          thickness=2,
                          lineType=cv2.LINE_AA)

            text = f'{class_name}:{per_score:.3f}'
            text_size = cv2.getTextSize(text, 0, 0.5, thickness=1)[0]
            fill_right_bottom = (max(left_top[0] + text_size[0], right_bottom[0]),
                                 left_top[1] - text_size[1] - 3)
            cv2.rectangle(origin_img,
                          left_top,
                          fill_right_bottom,
                          color=class_color,
                          thickness=-1,
                          lineType=cv2.LINE_AA)
            cv2.putText(origin_img,
                        text, (left_top[0], left_top[1] - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color=(0, 0, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)
        frame_num += 1
        print(f"processing frame: {frame_num}")

        if cfgs["save_video_path"]:
            out_video.write(origin_img)

        if cfgs["show_video"]:
            cv2.imshow('detection_result', origin_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    return


def test_video():
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps = ', fps)

    # 总帧数
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total fps = ', totalFrameNumber)

    while (True):
        # cap.read()函数返回的第1个参数ret是一个布尔值, 表示当前这一帧是否获取正确
        ret, frame = cap.read()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
        if c == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_p = r"D:\workspace\data\dl\test_images\1.mp4"
    # video_p = 0
    cfgs_dict = {
        "trained_dataset_name": "COCO",
        "model": "resnet50_retinanet",
        "video_path": video_p,
        "image_resize_style": "yolostyle",
        "input_image_size": 640,
        "decoder": "RetinaDecoder",
        "seed": 0,
        "trained_num_classes": 80,
        "trained_model_path": r"D:\Desktop\resnet50_retinanet-coco-yoloresize640-metric33.493.pth",
        "min_score_threshold": 0.5,
        "save_video_path": r"D:\Desktop\shows",
        "show_video": True
    }
    inference(cfgs_dict)
    # test_video()
