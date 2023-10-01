import os
import math
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.backends import cudnn
import json

import models
import decodes
from models.decoder import RetinaDecoder
from utils.util import mkdir_if_missing, load_state_dict, compute_macs_and_params
from datasets.voc import VOC_CLASSES, VOC_CLASSES_COLOR


class InferResizer:
    def __init__(self, resize=640, divisor=32):
        self.resize = resize
        self.divisor = 32

    def __call__(self, image):
        h, w, c = image.shape
        size = np.array([h, w]).astype(np.float32)
        scale = self.resize / max(h, w)
        resize_h, resize_w = math.ceil(h * scale), math.ceil(w * scale)

        resize_img = cv2.resize(image, (resize_w, resize_h))
        pad_w = 0 if resize_w % self.divisor == 0 else self.divisor - resize_w % self.divisor
        pad_h = 0 if resize_h % self.divisor == 0 else self.divisor - resize_h % self.divisor

        padded_img = np.zeros((resize_h + pad_h, resize_w + pad_w, 3),
                              dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = resize_img

        return {'image': padded_img, 'scale': np.array(scale).astype(np.float32), 'size': size}


def infer_folder():
    img_root = r'D:\workspace\data\dl\test_images\*.jpg'
    model_path = r'D:\Desktop\resnet50_retinanet-voc-yoloresize640-metric80.674.pth'
    save_root = r'D:\Desktop\shows'

    mkdir_if_missing(save_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_resizer = InferResizer(resize=640)
    dataset_name = 'voc'
    if dataset_name == 'voc':
        with open('./datasets/pascal_voc.json', 'r', encoding='utf-8') as fr:
            infos = json.load(fr)
        cat2idx = infos['class']
        idx2color = infos['color']
        idx2cat = {v: k for k, v in cat2idx.items()}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 20
    elif dataset_name == 'coco2017':
        with open('datasets/others/coco_classes.json', 'r', encoding='utf-8') as fr:
            infos = json.load(fr)
        cat2idx = infos['class']
        idx2color = infos['colors']
        idx2cat = {v: k for k, v in cat2idx.items()}
        mean = [0.471, 0.448, 0.408]
        std = [0.234, 0.239, 0.242]
        num_classes = 80
    else:
        raise ValueError(f'Unsuppoerted {dataset_name} type')

    mean = torch.tensor(mean, dtype=torch.float32, device=device).tile(1, 1, 1, 1)
    std = torch.tensor(std, dtype=torch.float32, device=device).tile(1, 1, 1, 1)

    infer_batch = 1
    cudnn.benchmark = True
    cudnn.deterministic = False

    use_gpu = True

    model = models.init_model(name='resnet50_retinanet',
                              num_classes=num_classes)
    model = load_state_dict(saved_model_path=model_path, model=model)
    if use_gpu:
        model = model.cuda()

    # checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # model.load_state_dict(checkpoint)
    model.eval()

    decoder = RetinaDecoder(min_score_threshold=0.5,
                            nms_type='diou_python_nms')
    img_lists = glob.glob(img_root)
    img_spilt_lists = [img_lists[start: start + infer_batch] for start in range(0, len(img_lists), infer_batch)]
    print(len(img_spilt_lists))

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    dummy_input = torch.rand(1, 3, 640, 640).to(device)
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    times = torch.zeros(len(img_spilt_lists))  # 存储每轮的时间

    font_size = 16
    font = ImageFont.truetype("./datasets/simhei.ttf", size=font_size)

    with torch.no_grad():
        for idx, img_spilt_list in enumerate(tqdm(img_spilt_lists)):
            images_src = [cv2.imread(p) for p in img_spilt_list]
            images_src = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in images_src]
            images_name = [p.split(os.sep)[-1] for p in img_spilt_list]
            images = []
            scales = []
            sizes = []
            for img in images_src:
                infos = infer_resizer(img)
                images.append(torch.from_numpy(infos['image']))
                scales.append(infos['scale'])
                sizes.append(infos['size'])
            images_tensor = torch.stack(images, dim=0).to(device)
            images_tensor = images_tensor / 255.
            images_tensor = (images_tensor - mean) / std
            images_tensor = images_tensor.permute(0, 3, 1, 2).contiguous()
            if use_gpu:
                images_tensor = images_tensor.cuda()

            starter.record()
            outs_tuple = model(images_tensor)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[idx] = curr_time

            batch_scores, batch_classes, batch_pred_bboxes = decoder(outs_tuple)

            # 处理每张图片
            for scores, classes, pred_bboxes, img, img_name, scale, size in \
                    zip(batch_scores, batch_classes, batch_pred_bboxes, images_src, images_name, scales, sizes):

                image = Image.fromarray(img)
                draw = ImageDraw.Draw(image)

                mask = classes >= 0
                scores, classes, pred_bboxes = scores[mask], classes[mask], pred_bboxes[mask]

                # clip boxes
                pred_bboxes[:, 0] = np.maximum(pred_bboxes[:, 0], 0)
                pred_bboxes[:, 1] = np.maximum(pred_bboxes[:, 1], 0)
                pred_bboxes[:, 2] = np.minimum(pred_bboxes[:, 2], size[1])
                pred_bboxes[:, 3] = np.minimum(pred_bboxes[:, 3], size[0])

                for class_id, bbox, score in zip(classes, pred_bboxes, scores):
                    bbox = bbox / scale
                    score = round(score, 3)
                    category_name = idx2cat[int(class_id)]
                    text = category_name + ' ' + str(score)
                    chars_w, chars_h = font.getsize(text)
                    category_color = tuple(idx2color[str(int(class_id))])
                    draw.rectangle(bbox[:4], outline=category_color, width=2)  # 绘制预测框
                    draw.rectangle((bbox[0], bbox[1] - chars_h, bbox[0] + chars_w, bbox[1]),
                                   fill=category_color)  # 文本填充框
                    draw.text((bbox[0], bbox[1] - font_size), text, fill=(255, 255, 255), font=font)
                save_path = os.path.join(save_root, img_name)
                image.save(save_path)
            # break
    mean_time = times.mean().item()
    print("Inference time: {:.2f} ms, FPS: {:.2f} ".format(mean_time, 1000 / mean_time))


def main_video():
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


from models.backbones.network_blocks import SiLU


def show_func():
    import torch.nn.functional as F
    x = torch.range(start=-15, end=15, step=0.25)
    # y = F.sigmoid(x)
    # y = F.tanh(x)
    y1 = F.relu(x)

    f = SiLU()
    y = f(x)

    plt.plot(x.detach(), y.detach(), color='r')
    plt.plot(x.detach(), y1.detach(), color='b')
    # plt.imshow()
    plt.grid()
    plt.show()


def load_image(cfgs, divisor=32):
    assert cfgs["image_resize_style"] in ['retinastyle',
                                          'yolostyle'], 'wrong style!'
    image = cv2.imread(cfgs["image_path"])
    origin_image = image.copy()
    h, w, _ = image.shape

    # normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(
        np.float32) / np.float(255.)
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

    return padded_img, origin_image, scale


def inference(cfgs):
    assert cfgs["trained_dataset_name"] in ['COCO', 'VOC'], 'Unsupported dataset!'
    assert cfgs["model"] in models.__dict__.keys(), 'Unsupported model!'
    assert cfgs["decoder"] in decodes.__dict__.keys(), 'Unsupported decoder!'
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

    model.eval()

    macs, params = compute_macs_and_params(cfgs["input_image_size"], model)
    print(f'model: {cfgs["model"]}, macs: {macs}, params: {params}')

    resized_img, origin_img, scale = load_image(cfgs)
    resized_img = torch.tensor(resized_img)
    # if input size:[B,3,640,640]
    # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
    # cls_heads shape:[[B, 80, 80, 9, 80],[B, 40, 40, 9, 80],[B, 20, 20, 9, 80],[B, 10, 10, 9, 80],[B, 5, 5, 9, 80]]
    # reg_heads shape:[[B, 80, 80, 9, 4],[B, 40, 40, 9, 4],[B, 20, 20, 9, 4],[B, 10, 10, 9, 4],[B, 5, 5, 9, 4]]
    outputs = model(resized_img.permute(2, 0, 1).float().unsqueeze(0))

    # batch_scores shape:[batch_size,max_object_num]
    # batch_classes shape:[batch_size,max_object_num]
    # batch_bboxes shape[batch_size,max_object_num,4]
    scores, classes, boxes = decoder(outputs)

    boxes /= scale

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

    if cfgs["trained_dataset_name"] == 'COCO':
        dataset_classes_name = "COCO_CLASSES"
        dataset_classes_color = "COCO_CLASSES_COLOR"
    else:
        dataset_classes_name = VOC_CLASSES
        dataset_classes_color = VOC_CLASSES_COLOR

    # draw all pred boxes
    for per_score, per_class_index, per_box in zip(scores, classes, boxes):
        per_score = per_score.astype(np.float32)
        per_class_index = per_class_index.astype(np.int32)
        per_box = per_box.astype(np.int32)

        class_name, class_color = dataset_classes_name[
            per_class_index], dataset_classes_color[per_class_index]

        left_top, right_bottom = (per_box[0], per_box[1]), (per_box[2],
                                                            per_box[3])
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

    if cfgs["save_image_path"]:
        cv2.imwrite(os.path.join(cfgs["save_image_path"], 'my.jpg'), origin_img)

    if cfgs["show_image"]:
        cv2.namedWindow("detection_result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow('detection_result', origin_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    cfgs_dict = {
        "trained_dataset_name": "VOC",
        "model": "resnet50_retinanet",
        "decoder": "RetinaDecoder",
        "seed": 0,
        "trained_num_classes": 20,
        "trained_model_path": r"D:\Desktop\resnet50_retinanet-voc-yoloresize640-metric80.674.pth",
        "input_image_size": 640,
        "min_score_threshold": 0.5,
        "save_image_path": r"D:\Desktop\shows",
        "show_image": True
    }
