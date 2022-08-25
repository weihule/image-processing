import os
import sys
import glob
import cv2
import numpy as np
import warnings
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.backends import cudnn
import json
warnings.filterwarnings('ignore')

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from torch_detection.retinanet.retina_decode import RetinaDecoder
from network_files.retinanet_model import resnet50_retinanet


class InferResizer:
    def __init__(self, resize=640):
        self.resize = resize

    def __call__(self, image):
        h, w, c = image.shape
        scale = self.resize / max(h, w)
        resize_h, resize_w = int(round(scale*h)), int(round(scale*w))

        resize_img = cv2.resize(image, (resize_w, resize_h))
        padded_img = np.zeros((self.resize, self.resize, 3), dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = resize_img
        padded_img = torch.from_numpy(padded_img)

        return {'img': padded_img, 'scale': scale}


def infer_folder(mode):

    img_root1 = '/workshop/weihule/data/detection_data/test_images/*.jpg'
    img_root2 = 'D:\\workspace\\data\\dl\\test_images\\*.jpg'

    model_path1 = '/workshop/weihule/data/detection_data/retinanet/checkpoints/resnet50_retinanet-metric80.558.pth'
    model_path2 = 'D:\\workspace\\data\\detection_data\\retinanet\\checkpoints\\latest.pth'
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

    with open('../utils/pascal_voc_classes.json', 'r', encoding='utf-8') as fr:
        infos = json.load(fr)
        voc_name2id = infos['classes']
        voc_colors = infos['colors']
    voc_id2name = {v: k for k, v in voc_name2id.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infer_resizer = InferResizer()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean, dtype=torch.float32, device=device).tile(1, 1, 1, 1)
    std = torch.tensor(std, dtype=torch.float32, device=device).tile(1, 1, 1, 1)

    infer_batch = 2
    cudnn.benchmark = True
    cudnn.deterministic = False

    with torch.no_grad():
        model = resnet50_retinanet(num_classes=20).to(device)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        checkpoint = checkpoint['model_state_dict']

        model.load_state_dict(checkpoint)
        model.eval()
        decoder = RetinaDecoder(min_score_threshold=0.2,
                                nms_type='python_nms',
                                nms_threshold=0.15)

        img_lists = glob.glob(img_root)
        img_spilt_lists = [img_lists[start: start + infer_batch] for start in range(0, len(img_lists), infer_batch)]
        print(len(img_spilt_lists))
        for img_spilt_list in tqdm(img_spilt_lists):
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
            preds = model(images_tensor)

            [batch_scores, batch_classes, batch_pred_bboxes] = decoder(preds)

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


def main_video():
    cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps = ', fps)

    # 总帧数
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('total fps = ', totalFrameNumber)

    while(True):
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
    mode_type = 'local'  # company    autodl
    # infer_folder(mode=mode_type)

    main_video()

