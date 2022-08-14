import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.backends import cudnn
import json

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from torch_detection.retinanet.retina_decode import RetinaNetDecoder
from network_files.retinanet_model import resnet50_retinanet
from torch_detection.retinanet.config import Config


class InferResizer:
    def __init__(self, resize=Config.input_image_size):
        self.resize = resize

    def __call__(self, image):
        h, w, c = image.shape
        if h >= w:
            scale = self.resize / h
            resize_w = int(round(scale * w))
            resize_h = self.resize
        else:
            scale = self.resize / w
            resize_w = self.resize
            resize_h = int(round(scale * h))

        resize_img = cv2.resize(image, (resize_w, resize_h))
        padded_img = np.zeros((self.resize, self.resize, 3), dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = resize_img
        padded_img = torch.from_numpy(padded_img)

        return {'img': padded_img, 'scale': scale}


def main():
    img_root = '/workshop/weihule/data/detection_data/test_images/*.jpg'
    model_path = '/workshop/weihule/data/detection_data/retinanet/checkpoints/resnet50_retinanet-metric80.558.pth'
    save_root = 'infer_shows'

    with open('../utils/pascal_voc_classes.json', 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
        voc_name2id = infos['classes']
        voc_colors = infos['colors']
    voc_id2name = {v: k for k, v in voc_name2id.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_resizer = InferResizer()
    mean = torch.tensor([[[[0.471, 0.448, 0.408]]]], dtype=torch.float32).to(device)
    std = torch.tensor([[[[0.234, 0.239, 0.242]]]], dtype=torch.float32).to(device)

    infer_batch = 8
    cudnn.benchmark = True
    cudnn.deterministic = False

    with torch.no_grad():
        model = resnet50_retinanet(num_classes=20).to(device)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        # rep = [['p3', 'P3'], ['p4', 'P4'], ['p5', 'P5'], ['p6', 'P6'], ['p7', 'P7']]
        # rep = []
        # model_load = {}
        # for k, v in checkpoint['model_state_dict'].items():
        #     for p in rep:
        #         if p[0] in k:
        #             k = k.replace(p[0], p[1])
        #     model_load[k] = v

        # model_load = torch.load(model_path, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint)
        model.eval()
        decoder = RetinaNetDecoder(image_w=Config.input_image_size,
                                   image_h=Config.input_image_size)

        img_lists = glob.glob(img_root)
        img_spilt_lists = [img_lists[start: start + infer_batch] for start in range(0, len(img_lists), infer_batch)]
        for img_spilt_list in img_spilt_lists:
            images_src = [cv2.imread(p) for p in img_spilt_list]
            images_src = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in images_src]
            images_name = [p.split(os.sep)[-1] for p in img_spilt_list]
            images = []
            scales = []
            for img in images_src:
                infos = infer_resizer(img)
                images.append(infos['img'])
                scales.append(infos['scale'])
            images_tensor = torch.stack(images, dim=0).to(device)
            images_tensor = images_tensor / 255.
            images_tensor = (images_tensor - mean) / std
            images_tensor = images_tensor.permute(0, 3, 1, 2).contiguous()
            heads = model(images_tensor)
            cls_heads, reg_heads, batch_anchors =
            batch_scores, batch_classes, batch_pred_bboxes = decoder(cls_heads, reg_heads, batch_anchors)
            batch_scores, batch_classes, batch_pred_bboxes = \
                batch_scores.cpu().numpy(), batch_classes.cpu().numpy(), batch_pred_bboxes.cpu().numpy()

            # 处理每张图片
            for scores, classes, pred_bboxes, img, img_name, scale in \
                    zip(batch_scores, batch_classes, batch_pred_bboxes, images_src, images_name, scales):

                image = Image.fromarray(img)
                draw = ImageDraw.Draw(image)
                font_size = 16
                font = ImageFont.truetype("../utils/simhei.ttf", size=font_size)

                mask = classes >= 0
                scores, classes, pred_bboxes = scores[mask], classes[mask], pred_bboxes[mask]
                for class_id, bbox, score in zip(classes, pred_bboxes, scores):
                    bbox = bbox / scale

                    score = round(score, 3)
                    category_name = voc_id2name[int(class_id)]
                    text = category_name + ' ' + str(score)
                    chars_w, chars_h = font.getsize(text)
                    category_color = tuple(voc_colors[int(class_id)])
                    draw.rectangle(bbox[:4], outline=category_color, width=2)  # 绘制预测框
                    draw.rectangle((bbox[0], bbox[1] - chars_h, bbox[0] + chars_w, bbox[1]),
                                   fill=category_color)  # 文本填充框
                    draw.text((bbox[0], bbox[1] - font_size), text, fill=(255, 255, 255), font=font)
                save_path = os.path.join(save_root, img_name)
                image.save(save_path)
            # break


if __name__ == "__main__":
    main()
