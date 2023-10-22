import os
import sys
import cv2
import numpy as np
import errno

import onnxruntime

from datasets.coco import COCO_CLASSES, COCO_CLASSES_COLOR
import decodes


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = COCO_CLASSES_COLOR[cls_id]
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        print(f"color = {color} {type(color)} {type(color[0])}")
        # txt_bk_color = (color * 0.7)
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


class ONNXInfer(object):
    def __init__(self, onnx_file,
                 resized_w=640,
                 resized_h=640,
                 score_thr=0.3,
                 output_dir='',
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.onnx_file = onnx_file
        self.resized_w = resized_w
        self.resized_h = resized_h
        self.score_thr = score_thr
        self.output_dir = output_dir
        self.mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
        self.std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))

        self.onnx_session = onnxruntime.InferenceSession(onnx_file,
                                                         providers=["CPUExecutionProvider"])
        self.input_name = [self.onnx_session.get_inputs()[0].name]
        self.output_name = [self.onnx_session.get_outputs()[0].name]

        # self.decoder = YOLOXDecoder(input_shape=(resized_h, resized_w))
        self.decoder = decodes.__dict__['YOLOXDecoder'](
            **{"input_shape": (resized_h, resized_w)}
        )

    def __call__(self, image_path):
        origin_img = cv2.imread(image_path)
        img, ratio = self.preprocss(origin_img, (self.resized_h, self.resized_w))
        input_feed = self.get_input_feed(self.input_name, np.expand_dims(img, axis=0))
        # (1, 3549, 85)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]
        # (3, 6) 图中目标数量, 四个位置参数加置信度和类别
        dets = self.decoder(output, ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                            conf=self.score_thr, class_names=COCO_CLASSES)

        mkdir_if_missing(self.output_dir)
        output_path = os.path.join(self.output_dir, os.path.basename(image_path))
        cv2.imshow('win', origin_img)
        cv2.waitKey(0)
        # cv2.imwrite(output_path, origin_img)

    def preprocss(self, img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    @staticmethod
    def get_input_feed(input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
    

if __name__ == "__main__":
    oi = ONNXInfer(onnx_file=r"D:\workspace\data\training_data\yolox\yolox_nano.onnx",
                   resized_h=416,
                   resized_w=416,
                   output_dir=r'D:\Desktop')
    image_path_ = r"D:\workspace\data\test_images\001.jpg"
    oi(image_path=image_path_)

