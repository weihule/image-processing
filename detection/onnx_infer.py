import os
import sys
import cv2
import numpy as np
import errno

import onnxruntime

from datasets.coco import COCO_CLASSES, COCO_CLASSES_COLOR
import decodes


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version. 不区分类别"""
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


class YOLOXDecoder:
    def __init__(self, input_shape=(640, 640), p6=False):
        self.input_shape = input_shape
        self.p6 = p6

    def __call__(self, outputs, ratio):
        # (1, 3549, 85)
        predictions = self.postprocess(outputs)

        # 没必要再留着batch维度了, 因为模型最终都是用在视频上的, 也就是需要一帧一帧的预测
        predictions = predictions[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio

        # [3, 6] 即 图中目标数量, 四个位置参数加置信度和类别
        dets = multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)

        return dets

    def postprocess(self, outputs):
        grids = []
        expanded_strides = []
        strides = [8, 16, 32] if not self.p6 else [8, 16, 32, 64]

        hsizes = [self.input_shape[0] // stride for stride in strides]
        wsizes = [self.input_shape[1] // stride for stride in strides]

        for hsize, wsize, stride in zip(hsizes, wsizes, strides):
            xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            expanded_strides.append(np.full((*shape, 1), stride))

        grids = np.concatenate(grids, 1)
        expanded_strides = np.concatenate(expanded_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

        return outputs


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
                 output_dir=''):
        self.onnx_file = onnx_file
        self.resized_w = resized_w
        self.resized_h = resized_h
        self.score_thr = score_thr
        self.output_dir = output_dir

        self.onnx_session = onnxruntime.InferenceSession(onnx_file,
                                                         providers=["CUDAExecutionProvider"])
        self.input_name = [self.onnx_session.get_inputs()[0].name]
        self.output_name = [self.onnx_session.get_outputs()[0].name]

        # self.decoder = YOLOXDecoder(input_shape=(resized_h, resized_w))
        self.decoder = YOLOXDecoder(input_shape=(resized_h, resized_w))

    def __call__(self, image_info):
        # print(type(image_info))
        if isinstance(image_info, np.ndarray):
            origin_img = image_info
        else:
            origin_img = cv2.imread(image_info)
        print(origin_img.shape)
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
        
        return origin_img

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
    

def single_infer(onnx_file, resized_w, resized_h, output_dir, image_path):
    oi = ONNXInfer(onnx_file, resized_w, resized_h, output_dir=output_dir)
    oi(image_path)


def video_infer(onnx_file, resized_w, resized_h, output_dir):
    oi = ONNXInfer(onnx_file, resized_w, resized_h, output_dir=output_dir)
    capture = cv2.VideoCapture(0)  # capture=cv2.VideoCapture("1.mp4")
    # fps = 0.0
    while True:
        # 读取某一帧
        ref, frame = capture.read()
        img = oi(frame)
        cv2.imshow('win', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    onnx_file_ = r"D:\Desktop\yolox_nano.onnx"
    resized_h_ = 416
    resized_w_ = 416
    output_dir_ = r'D:\Desktop'
    
    image_path_ = r"D:\workspace\data\dl\test_images\001.jpg"
    video_infer(onnx_file_, resized_w_, resized_h_, output_dir_)
    # single_infer(onnx_file_, resized_w_, resized_h_, output_dir_, image_path_)

