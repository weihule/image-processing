import os
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path
import json
import time
import common

batch_size = 8
num_classes = 5
USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32


def get_cls_idx(class_file=None):
    # 加载类别索引和类别名称映射
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    with open(class_file, "r", encoding="utf-8") as fr:
        cls2idx = json.load(fr)
    
    return cls2idx


def softmax(x):
    """ softmax function """

    # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x -= np.max(x, axis=1, keepdims=True)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


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

    infer(context)


def infer(context):
    cls2idx = get_cls_idx()
    idx2cls = {v: k for k, v in cls2idx.items()}
    loader = batch_images_gen(image_dir=r"D:\workspace\data\dl\flower\test",
                              batch_size=batch_size,
                              img_w=224,
                              img_h=224,
                              mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225))
    stream = cuda.Stream()
    len_dataset = 0
    for images, image_infos in loader:
        len_dataset += len(image_infos)
        output = np.empty([batch_size, num_classes], dtype=target_dtype)

        d_input = cuda.mem_alloc(1 * images.nbytes)
        d_output = cuda.mem_alloc(1 * output.nbytes)
        bindings = [int(d_input), int(d_output)]

        # transfer input data to device
        cuda.memcpy_htod_async(d_input, images, stream)

        # execute model 此处采用异步推理。如果想要同步推理，需将execute_async_v2替换成execute_v2
        context.execute_async_v2(bindings, stream.handle, None)

        # transfer predictions back
        cuda.memcpy_dtoh_async(output, d_output, stream)

        # syncronize threads
        stream.synchronize()

        preds = softmax(output)
        scores, indices = np.max(preds, axis=-1), np.argmax(preds, axis=-1)
        for ii, s, i in zip(image_infos, scores, indices):
            print(f"file_name: {ii.name} pred_class: {idx2cls[int(i)]} pred_score: {s}")


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


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            config.max_workspace_size = 1 << 28  # 256MiB
            builder.max_batch_size = 8
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            # Dynamic input setting
            profile = builder.create_optimization_profile()
            profile.set_shape("input", (8, 3, 224, 224), (8, 3, 400, 400), (8, 3, 512, 512))
            config.add_optimization_profile(profile)

            # network.get_input(0).shape = [1, 3, 608, 608]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            # plan = builder.build_serialized_network(network, config)
            # engine = runtime.deserialize_cuda_engine(plan)
            engine = builder.build_engine(network, config)

            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                # f.write(plan)
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


if __name__ == "__main__":
    onnx_file = r"D:\workspace\data\training_data\resnet50\pths\resnet50-0.934.onnx"
    engine_file = r"D:\workspace\data\training_data\resnet50\pths\resnet_engine2.trt"
    # main()
    get_engine(onnx_file_path=onnx_file,
               engine_file_path=engine_file)

