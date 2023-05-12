import sys
import onnx
import onnxruntime
import json
import numpy as np
import cv2
import time


class ONNXInfer(object):
    def __init__(self, onnx_file,
                 cls2name,
                 resized_w=224,
                 resized_h=224,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 batch_size=6):
        self.onnx_file = onnx_file
        self.cls2name = cls2name
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
        score, index, name = self.post_process(pred, self.cls2name)

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
    def post_process(pred, cls2name):
        pred = softmax(pred)
        score = np.max(pred, axis=-1)[0]
        index = np.argmax(pred, axis=-1)[0]

        return score, index, cls2name[str(index)].split(",")[0]

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
    with open("./utils/imagenet1000.json", "r", encoding="utf-8") as fr:
        cls2name = json.load(fr)
    img_path = r"D:\workspace\data\dl\imagenet100\imagenet100_val\African-hunting-dog\ILSVRC2012_val_00000078.jpg"
    onnx_file_ = "D:\\Desktop\\resnet50-acc76.264.onnx"
    oi = ONNXInfer(onnx_file=onnx_file_,
                   cls2name=cls2name)

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


if __name__ == "__main__":
    main()


