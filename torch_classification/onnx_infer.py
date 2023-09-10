import os
import sys
import onnx
from pathlib import Path
import onnxruntime
import json
import numpy as np
import cv2
import time


class ONNXInfer(object):
    def __init__(self, onnx_file,
                 cls2idx,
                 resized_w=224,
                 resized_h=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 batch_size=6):
        self.onnx_file = onnx_file
        self.cls2idx = cls2idx
        self.resized_w = resized_w
        self.resized_h = resized_h
        self.mean = np.asarray(mean, dtype=np.float32).reshape((1, 1, 1, 3))
        self.std = np.asarray(std, dtype=np.float32).reshape((1, 1, 1, 3))
        self.batch_size = batch_size

        self.onnx_session = onnxruntime.InferenceSession(onnx_file,
                                                         providers=["CPUExecutionProvider"])
        self.input_name = [self.onnx_session.get_inputs()[0].name]
        self.output_name = [self.onnx_session.get_outputs()[0].name]

    def infer(self, image):
        pred = self.prepare_single(image)
        score, index, name = self.post_process(pred, self.cls2idx)

        return score, index, name

    def prepare_single(self, image):
        """
        处理单帧图像
        """
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        image = (image / 255. - self.mean) / self.std
        image = image.transpose(0, 3, 1, 2)

        input_feed = self.get_input_feed(self.input_name, image)
        pred = self.onnx_session.run(self.output_name, input_feed=input_feed)[0]

        return pred

    @staticmethod
    def post_process(pred, cls2idx):
        pred = softmax(pred)
        score = np.max(pred, axis=-1)[0]
        index = np.argmax(pred, axis=-1)[0]

        return score, index, cls2idx[str(index)].split(",")[0]

    @staticmethod
    def get_input_feed(input_name, image_numpy):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed


def softmax(x):
    """ softmax function """

    # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x -= np.max(x, axis=1, keepdims=True)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


def main():
    # 加载类别索引和类别名称映射
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    with open(class_file, "r", encoding="utf-8") as fr:
        cls2idx = json.load(fr)
    img_path = r"D:\workspace\data\dl\imagenet100\imagenet100_val\African-hunting-dog\ILSVRC2012_val_00000078.jpg"
    onnx_file_ = r"D:\workspace\data\training_data\resnet50\pths\resnet50-0.934.onnx"
    oi = ONNXInfer(onnx_file=onnx_file_,
                   cls2idx=cls2idx)

    capture = cv2.VideoCapture(0)
    while True:
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()

        # 进行检测
        score, index, name = oi.infer(frame)
        fps = 1. / (time.time() - t1)
        print("fps= %.2f" % fps)
        frame = cv2.putText(frame, f"fps= {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.putText(frame, f"{name:<15s} {score:.2f}", (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("video", frame)
        # 按下esc键，退出
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break


class OnnxInfer:
    def __init__(self,
                 image_dir,
                 onnx_file,
                 cls2idx,
                 img_w=224,
                 img_h=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 batch_size=6,
                 provider="cpu"):
        self.image_dir = image_dir
        self.onnx_file = onnx_file
        self.cls2idx = cls2idx
        self.idx2cls = {v: k for k, v in cls2idx.items()}
        self.img_w, self.img_h = img_w, img_h
        self.mean, self.std = mean, std
        self.batch_size = batch_size

        if provider == "cpu":
            self.onnx_session = onnxruntime.InferenceSession(onnx_file,
                                                             providers=["CPUExecutionProvider"])
        else:
            self.onnx_session = onnxruntime.InferenceSession(onnx_file,
                                                             providers=["CUDAExecutionProvider"])
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

    def batch_images_gen(self):
        """
        batch_images生成器
        """
        try:
            images_path = []
            for i in Path(self.image_dir).iterdir():
                images_path.append(i)
            batch_images_path = [images_path[s: s+self.batch_size] for
                                 s in range(0, len(images_path), self.batch_size)]
            for bis in batch_images_path:
                batch_image = []
                for per_img_path in bis:
                    image = cv2.imread(str(per_img_path))
                    image = cv2.resize(image, (self.img_w, self.img_h))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    batch_image.append(image)
                # [B, H, W, C]
                batch_image = np.stack(batch_image, axis=0)
                # 归一化和标准化
                batch_image = ((batch_image / 255.) - self.mean) / self.std
                # [B, C, H, W]
                batch_image = np.transpose(batch_image, (0, 3, 1, 2)).astype(np.float32)
                yield batch_image, bis
        except FileNotFoundError:
            print("文件不存在或路径错误")
        except Exception as e:
            print(f"发生了其他异常: {e}")

    def infer(self):
        loader = self.batch_images_gen()
        for images, image_infos in loader:
            input_feed = {self.input_name: images}
            preds = self.onnx_session.run([self.output_name], input_feed)
            preds = preds[0]
            preds = softmax(preds)
            scores, indices = np.max(preds, axis=-1), np.argmax(preds, axis=-1)
            for ii, s, i in zip(image_infos, scores, indices):
                print(f"file_name: {ii.name} pred_class: {self.idx2cls[int(i)]} pred_score: {s}")


def run():
    # onnx权重文件路径
    onnx_file = r"D:\workspace\data\training_data\resnet50\pths\resnet50-0.934.onnx"

    # 加载类别索引和类别名称映射
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    with open(class_file, "r", encoding="utf-8") as fr:
        cls2idx = json.load(fr)

    # 存放测试图片的文件夹路径
    image_dir = r"D:\workspace\data\dl\flower\test"
    oi = OnnxInfer(onnx_file=onnx_file,
                   cls2idx=cls2idx,
                   image_dir=image_dir,
                   batch_size=8,
                   provider="cuda")
    oi.infer()


if __name__ == "__main__":
    # main()
    run()
    print(onnxruntime.get_device())
    # print(onnxruntime.get_available_providers())
