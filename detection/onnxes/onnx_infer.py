import os
import numpy as np
import onnxruntime
import json
from PIL import Image, ImageFont, ImageDraw
import cv2
from tqdm import tqdm
from onnx_decoder import RetinaDecoder
import warnings
warnings.filterwarnings('ignore')


class InferResizer:
    def __init__(self, resize=420):
        self.resize = resize

    def __call__(self, image):
        h, w, _ = image.shape
        size = np.array([h, w]).astype(np.float32)
        scale = self.resize / max(h, w)
        resize_h, resize_w = int(round(scale * h)), int(round(scale * w))
        resize_img = cv2.resize(image, (resize_w, resize_h))
        padded_img = np.zeros((self.resize, self.resize, 3), dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = resize_img

        return {'img': padded_img, 'scale': np.array(scale).astype(np.float32), 'size': size}


class ONNXModel:
    def __init__(self, onnx_path, resize):
        """
        :param onnx_path:
        """
        # model_file = open(onnx_path, 'rb')
        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)

        self.decoder = RetinaDecoder(min_score_threshold=0.2,
                                     nms_type='python_nms',
                                     nms_threshold=0.5)
        self.dataset_name = 'coco2017'
        self.batch_infer = 2
        self.infer_resizer = InferResizer(resize=resize)
        if self.dataset_name == 'voc':
            with open('../datasets/others/pascal_voc_classes.json', 'r', encoding='utf-8') as fr:
                infos = json.load(fr)
                name2id = infos['classes']
                self.colors = infos['colors']
            self.id2name = {v: k for k, v in name2id.items()}
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1, 1, 1, -1))
            self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1, 1, 1, -1))
        elif self.dataset_name == 'coco2017':
            with open('../datasets/others/coco_classes.json', 'r', encoding='utf-8') as fr:
                infos = json.load(fr)
                name2id = infos['COCO_CLASSES']
                self.colors = infos['colors']
            self.id2name = {v: k for k, v in name2id.items()}
            self.mean = np.array([0.471, 0.448, 0.408], dtype=np.float32).reshape((1, 1, 1, -1))
            self.std = np.array([0.234, 0.239, 0.242], dtype=np.float32).reshape((1, 1, 1, -1))
        else:
            raise ValueError(f'Unsuppoerted {self.dataset_name} type')

        self.save_root = 'D:\\Desktop\\tempfile\\shows'

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def prepare(self, image_paths):
        img_path_splits = [image_paths[start: start + self.batch_infer]
                           for start in range(0, len(image_paths), self.batch_infer)]
        all_images_src = []
        all_images_numpy = []
        all_images_name = []
        all_images_scale = []
        all_images_size = []
        for idx, img_path_split in enumerate(tqdm(img_path_splits)):
            images_src = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in img_path_split]
            images_name = [p.split(os.sep)[-1] for p in img_path_split]
            images = []
            scales = []
            sizes = []
            for img in images_src:
                infos = self.infer_resizer(img)
                images.append(infos['img'])
                scales.append(infos['scale'])
                sizes.append(infos['size'])
            images_numpy = np.stack(images, axis=0)     # [batch_size, h, w, 3]
            images_numpy = ((images_numpy / 255.) - self.mean) / self.std
            images_numpy = images_numpy.transpose(0, 3, 1, 2)

            all_images_src.append(images_src)
            all_images_numpy.append(images_numpy)
            all_images_name.append(images_name)
            all_images_scale.append(scales)
            all_images_size.append(sizes)

        return all_images_src, all_images_numpy, all_images_name, all_images_scale, all_images_size

    def process(self, prepares):
        all_images_src, all_images_numpy, all_images_name, all_images_scale, all_images_size = prepares
        for images_src, images_numpy, images_name, images_scale, images_size in zip(
                all_images_src, all_images_numpy, all_images_name, all_images_scale, all_images_size):
            input_feed = self.get_input_feed(self.input_name, images_numpy)
            output_value = self.onnx_session.run(self.output_name, input_feed=input_feed)
            # for i in output_value:
            #     print('===', i.shape)
            batch_scores, batch_classes, batch_pred_bboxes = self.decoder(output_value)

            # 处理每张图片
            for img, scores, classes, pred_bboxes, img_name, scale, size in zip(
                    images_src, batch_scores, batch_classes, batch_pred_bboxes, images_name, images_scale, images_size):
                image = Image.fromarray(img)
                draw = ImageDraw.Draw(image)
                font_size = 16
                font = ImageFont.truetype("../utils/simhei.ttf", size=font_size)

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
                    category_name = self.id2name[int(class_id)]
                    text = category_name + ' ' + str(score)
                    chars_w, chars_h = font.getsize(text)
                    category_color = tuple(self.colors[int(class_id)])
                    draw.rectangle(bbox[:4], outline=category_color, width=2)  # 绘制预测框
                    draw.rectangle((bbox[0], bbox[1] - chars_h, bbox[0] + chars_w, bbox[1]),
                                   fill=category_color)  # 文本填充框
                    draw.text((bbox[0], bbox[1] - font_size), text, fill=(255, 255, 255), font=font)
                save_path = os.path.join(self.save_root, img_name)
                image.save(save_path)

    def __call__(self, image_paths):
        prepares = self.prepare(image_paths)
        self.process(prepares)


if __name__ == '__main__':
    import glob
    onnx_model = 'D:\\Desktop\\tempfile\\best_model.onnx'
    img_root = 'D:\\workspace\data\\dl\\test_images\\*.jpg'
    img_paths = glob.glob(img_root)
    om = ONNXModel(onnx_model, resize=420)
    om(img_paths)

