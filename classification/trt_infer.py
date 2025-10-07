import os
import cv2
import json
import onnx
import time
import random
from pathlib import Path

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

use_float16 = False
target_type = np.float16 if use_float16 else np.float32


def batch_images_gen(image_dir, batch_size, img_w, img_h, mean, std, shuffle=True):
    """
    batch_images生成器
    """
    try:
        images_path = []
        for i in Path(image_dir).iterdir():
            images_path.append(i)
        if shuffle:
            random.shuffle(images_path)
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
            batch_image = np.ascontiguousarray(batch_image).astype(target_type)
            yield batch_image, bis
    except FileNotFoundError:
        print("文件不存在或路径错误")
    except Exception as e:
        print(f"发生了其他异常: {e}")


def get_engine(onnx_file_path, engine_file_path):
    # 定义TensorRT Logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        # 创建TensorRT Builder
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:

            # 指定 FP16 精度
            config.set_flag(trt.BuilderFlag.FP16)

            # 解析ONNX模型
            with open(onnx_file_path, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    exit(1)

            # 创建优化配置文件
            profile = builder.create_optimization_profile()
            profile.set_shape('input', (1, 3, 224, 224), (8, 3, 224, 224), (16, 3, 224, 224))  # 指定输入维度范围
            config.add_optimization_profile(profile)

            # 启用 FP16 精度
            # if use_float16:
            #     config.set_flag(trt.BuilderFlag.FP16)

            # 构建TensorRT引擎
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)

            # 保存引擎到文件
            with open(engine_file_path, "wb") as engine_file:
                engine_file.write(engine.serialize())

            return engine


def main(onnx_file, engine_file):
    # 加载类别索引和类别名称映射
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    with open(class_file, "r", encoding="utf-8") as fr:
        cls2idx = json.load(fr)
    idx2cls = {v: k for k, v in cls2idx.items()}
    # 设置最大和最小批处理大小
    batch_size = 8
    num_classes = 5
    # 指定输入形状，不包括批处理大小
    input_shape = (3, 224, 224)

    loader = batch_images_gen(image_dir=r"D:\workspace\data\dl\flower\test",
                              batch_size=8,
                              img_w=224,
                              img_h=224,
                              mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225))
    engine = get_engine(onnx_file, engine_file)
    len_dataset = 0
    start_time = time.time()
    # 创建CUDA上下文
    with engine.create_execution_context() as context:
        for images, image_infos in loader:
            len_dataset += len(image_infos)
            input_data = images

            d_input = cuda.mem_alloc(input_data.nbytes)
            cuda.memcpy_htod(d_input, input_data)

            output = np.empty((batch_size, num_classes), dtype=target_type)
            d_output = cuda.mem_alloc(output.nbytes)

            # 设置动态批处理大小
            # context.active_optimization_profile = 0  # 使用第一个优化配置(8.x已弃用)
            # context.set_optimization_profile_async(0)   # 使用第一个优化配置
            context.set_binding_shape(0, (batch_size, *input_shape))  # 设置输入的动态形状

            # 执行推理
            context.execute_v2(bindings=[int(d_input), int(d_output)])

            # 在这里处理输出数据
            output_data = np.empty((batch_size, num_classes), dtype=target_type)
            # 从 GPU 内存中获取输出
            cuda.memcpy_dtoh(output_data, d_output)

            preds = softmax(output_data)
            scores, indices = np.max(preds, axis=-1), np.argmax(preds, axis=-1)
            for ii, s, i in zip(image_infos, scores, indices):
                print(f"file_name: {ii.name:20s} pred_class: {idx2cls[int(i)]:10s} pred_score: {s}")
            # break

        cost_time = time.time() - start_time
        print(f"inference time: {cost_time:.2f} s  numbers: {len_dataset}  fps: {1 / (cost_time / len_dataset):.2f}")


def softmax(x):
    """ softmax function """

    # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x -= np.max(x, axis=1, keepdims=True)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


if __name__ == "__main__":
    o_file = r"D:\workspace\data\training_data\resnet50\pths\resnet50-0.9421.onnx"
    e_file = r"D:\workspace\data\training_data\resnet50\pths\resnet_engine3.trt"
    main(o_file, e_file)


