import os
import sys
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.backends import cudnn
import json

import models
from models.decoder import RetinaDecoder
from util.utils import mkdir_if_missing

warnings.filterwarnings('ignore')


class InferResizer:
    def __init__(self, resize=640):
        self.resize = resize

    def __call__(self, image):
        h, w, c = image.shape
        size = np.array([h, w]).astype(np.float32)
        scale = self.resize / max(h, w)
        resize_h, resize_w = int(round(scale * h)), int(round(scale * w))

        resize_img = cv2.resize(image, (resize_w, resize_h))
        padded_img = np.zeros((self.resize, self.resize, 3), dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = resize_img
        padded_img = torch.from_numpy(padded_img)

        return {'img': padded_img, 'scale': np.array(scale).astype(np.float32), 'size': size}


def infer_folder(mode):
    img_root1 = '/workshop/weihule/data/detection_data/test_images/*.jpg'
    img_root2 = 'D:\\workspace\\data\\dl\\test_images\\*.jpg'

    model_path1 = '/workshop/weihule/data/detection_data/retinanet/checkpoints/resnet50_retinanet-metric80.558.pth'
    model_path2 = 'D:\\Desktop\\tempfile\\best_model.pth'
    model_path3 = '/workshop/weihule/data/detection_data/retinanet/checkpoints/latest.pth'

    save_root1 = 'D:\\Desktop\\shows'
    save_root2 = '/workshop/weihule/code/study/torch_detection/retinanet/infer_shows'

    if mode == 'local':
        img_root = img_root2
        model_path = model_path2
        save_root = save_root1
    elif mode == 'company':
        img_root = img_root1
        model_path = model_path3
        save_root = save_root2
    else:
        raise 'wrong value'
    mkdir_if_missing(save_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_resizer = InferResizer(resize=400)
    dataset_name = 'coco2017'
    if dataset_name == 'voc':
        with open('./datasets/pascal_voc_classes.json', 'r', encoding='utf-8') as fr:
            infos = json.load(fr)
            name2id = infos['classes']
            colors = infos['colors']
        id2name = {v: k for k, v in name2id.items()}
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 20
    elif dataset_name == 'coco2017':
        with open('./datasets/coco_classes.json', 'r', encoding='utf-8') as fr:
            infos = json.load(fr)
            name2id = infos['COCO_CLASSES']
            colors = infos['colors']
        id2name = {v: k for k, v in name2id.items()}
        mean = [0.471, 0.448, 0.408]
        std = [0.234, 0.239, 0.242]
        num_classes = 80
    else:
        raise ValueError(f'Unsuppoerted {dataset_name} type')
    mean = torch.tensor(mean, dtype=torch.float32, device=device).tile(1, 1, 1, 1)
    std = torch.tensor(std, dtype=torch.float32, device=device).tile(1, 1, 1, 1)

    infer_batch = 2
    cudnn.benchmark = True
    cudnn.deterministic = False

    use_gpu = True

    model = models.init_model(name='resnet50_retinanet',
                              num_classes=num_classes,
                              pre_train_load_dir=None
                              )
    if use_gpu:
        model = model.cuda()

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    # checkpoint = checkpoint['model_state_dict']

    model.load_state_dict(checkpoint)
    model.eval()

    decoder = RetinaDecoder(min_score_threshold=0.21,
                            nms_type='diou_python_nms')
    img_lists = glob.glob(img_root)
    img_spilt_lists = [img_lists[start: start + infer_batch] for start in range(0, len(img_lists), infer_batch)]
    print(len(img_spilt_lists))

    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    dummy_input = torch.rand(1, 3, 256, 256).to(device)
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    times = torch.zeros(len(img_spilt_lists))   # # 存储每轮的时间

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
                images.append(infos['img'])
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
                font_size = 16
                font = ImageFont.truetype("./utils/simhei.ttf", size=font_size)

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
                    category_name = id2name[int(class_id)]
                    text = category_name + ' ' + str(score)
                    chars_w, chars_h = font.getsize(text)
                    category_color = tuple(colors[int(class_id)])
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


def show_func():
    import torch.nn.functional as F
    x = torch.arange(start=-15, end=15)
    # y = F.sigmoid(x)
    # y = F.tanh(x)
    y = F.relu(x)

    plt.plot(x.detach(), y.detach())
    # plt.imshow()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    mode_type = 'local'  # company    autodl
    # infer_folder(mode=mode_type)

    show_func()


