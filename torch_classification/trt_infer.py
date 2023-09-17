import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import json
import time

BATCH_SIZE = 8
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32

def get_cls_idx(class_file=None):
    # 加载类别索引和类别名称映射
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    with open(class_file, "r", encoding="utf-8") as fr:
        cls2idx = json.load(fr)
    
    return cls2idx


def batch_images_gen(image_dir, batch_size, img_w, img_h, mean, std):
    """
    batch_images生成器
    """
    try:
        images_path = []
        for i in Path(image_dir).iterdir():
            images_path.append(i)
        batch_images_path = [images_path[s: s+batch_size] for
                             s in range(0, len(images_path), batch_size)]
        for bis in batch_images_path:
            batch_image = []
            for per_img_path in bis:
                image = cv2.imread(str(per_img_path))
                image = cv2.resize(image, (img_w, img_h))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                batch_image.append(image)
            # [B, H, W, C]
            batch_image = np.stack(batch_image, axis=0)
            # 归一化和标准化
            batch_image = ((batch_image / 255.) - mean) / std
            # [B, C, H, W]
            batch_image = np.transpose(batch_image, (0, 3, 1, 2))
            batch_image = np.ascontiguousarray(batch_image).astype(target_dtype)
            yield batch_image, bis
    except FileNotFoundError:
        print("文件不存在或路径错误")
    except Exception as e:
        print(f"发生了其他异常: {e}")

def main():
    trt_path = r"D:\workspace\data\training_data\resnet50\pths\resnet_engine.trt"

    # 创建Runtime, 加载trt引擎
    f = open(trt_path, "rb")  # 读取trt模型
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))  # 创建一个Runtime(传入记录器Logger)
    engine = runtime.deserialize_cuda_engine(f.read())  # 从文件中加载trt引擎
    context = engine.create_execution_context()  # 创建context

    # 分配input和output内存
    input_batch = np.random.randn(BATCH_SIZE, 224, 224, 3).astype(target_dtype)
    output = np.empty([BATCH_SIZE, 1000], dtype=target_dtype)

    d_input = cuda.mem_alloc(1 * input_batch.nbytes)
    d_output = cuda.mem_alloc(1 * output.nbytes)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()


def infer(context):
    loader = batch_images_gen(image_dir=r"D:\workspace\data\dl\flower\test",
                              batch_size=BATCH_SIZE,
                              img_w=224,
                              img_h=224,
                              mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225))
    for images, image_infos in loader:


# 5. 创建predict函数
def predict(batch, d_input, stream, context, bindings, d_output, output):  # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)  # 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # syncronize threads
    stream.synchronize()

    return output


def preprocess_input(inputs):      # input_batch无法直接传给模型，还需要做一定的预处理
    # 此处可以添加一些其它的预处理操作（如标准化、归一化等）
    result = np.transpose(inputs, (0, 3, 1, 2))
    result = np.ascontiguousarray(result, dtype=target_dtype)
    return result.astype(target_dtype)


preprocessed_inputs = preprocess_input(input_batch)
print("====", preprocessed_inputs.shape)

print("Warming up...")
pred = predict(preprocessed_inputs)
print("Done warming up!")

t0 = time.time()
pred2 = predict(preprocessed_inputs)
t = time.time() - t0
print("Prediction cost {:.4f}s".format(t))

print(trt.__version__)


class TRTInfer:
    def __init__(self, trt_path):

