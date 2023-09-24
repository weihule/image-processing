import os
import cv2
import json
import onnx
import time
from pathlib import Path

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
USE_FP16 = False
target_dtype = np.float16 if USE_FP16 else np.float32


def get_engine(onnx_file_path, engine_file_path=""):
    def build_engine():
        """
        Takes an ONNX file and creates a TensorRT engine to run inference with
        """
        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(EXPLICIT_BATCH) as network, \
                builder.create_builder_config() as config, \
                trt.OnnxParser(network, TRT_LOGGER) as parser, \
                trt.Runtime(TRT_LOGGER) as runtime:
            # config.max_workspace_size = 1 << 28     # 256MiB
            # builder.max_batch_size = 1
            config.set_memory_pool_limit(trt.MemoryPoolType.DLA_LOCAL_DRAM, 256 * 1024 * 1024)  # 256MiB
            # parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found ".format(onnx_file_path)
                )
                exit(0)
            print("loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            profile = builder.create_optimization_profile()
            # Dynamic input setting
            profile.set_shape("input", (1, 3, 224, 224), (1, 3, 512, 512), (1, 3, 1024, 1024))
            config.add_optimization_profile(profile)
            # network.get_input(0).shape = [1, 3, 224, 224]
            print("completed parsing of onnx file")
            print(f"building an engine from file {onnx_file_path}; this may take a while ...")
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def process_image(img_path):
    mean = (0.485, 0.456, 0.406),
    std = (0.229, 0.224, 0.225)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))
    # [H, W, C]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # [1, H, W, C]
    image = np.expand_dims(image, axis=0)
    image = ((image / 255.) - mean) / std
    # [B, C, H, W]
    image = np.transpose(image, (0, 3, 1, 2))
    image = np.ascontiguousarray(image).astype(target_dtype)

    return image


def main():
    # 加载类别索引和类别名称映射
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    with open(class_file, "r", encoding="utf-8") as fr:
        cls2idx = json.load(fr)
    idx2cls = {v: k for k, v in cls2idx.items()}

    # 定义函数加载 TensorRT 引擎
    def load_trt_engine(trt_file_path):
        with open(trt_file_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    # 加载保存的 TensorRT 引擎
    trt_file_path = r"D:\workspace\data\training_data\resnet50\pths\resnet_engine2.trt"
    trt_engine = load_trt_engine(trt_file_path)

    # 创建一个 CUDA 上下文
    cuda_context = trt_engine.create_execution_context()

    # 准备输入数据（根据模型的输入要求准备数据）
    input_data = process_image(img_path=r"D:\workspace\data\dl\flower\test\tulips11.jpg").astype(np.float32)

    # 在 GPU 上分配内存并传输输入数据
    d_input = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)

    # 创建输出数据的 GPU 内存
    output = np.empty((1, 5), dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)

    # 执行推理
    cuda_context.execute_v2(bindings=[int(d_input), int(d_output)])

    # 将输出数据从 GPU 拷贝到主机内存
    cuda.memcpy_dtoh(output, d_output)

    # 输出推理结果
    print("推理结果:", output)

    preds = softmax(output)
    scores, indices = np.max(preds, axis=-1), np.argmax(preds, axis=-1)
    print(scores, indices, idx2cls[int(indices)])


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


def get_engine2(onnx_file_path, engine_file_path):
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, \
                trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        # 创建一个网络 network 并启用显式批处理
        # 创建一个 Builder 配置对象 config，它可以用来配置 TensorRT 构建过程的各种选项。
        # 创建一个 ONNX 解析器对象 parser，并将其与网络 network 和日志记录器 TRT_LOGGER 关联
        # 创建 TensorRT 运行时对象 runtime，它将用于执行 TensorRT 引擎。
        # 加载 ONNX 模型
        onnx_model = onnx.load(onnx_file_path)

        # 创建 TensorRT Builder、Network 和 Config
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()

        # config.max_workspace_size = 1 << 28  # 设置工作空间大小
        config.set_memory_pool_limit(1 << 28)  # 设置工作空间大小为256MiB

        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # 使用 ONNX 解析器将 ONNX 模型解析到 TensorRT 网络中
        parser = trt.OnnxParser(network, TRT_LOGGER)
        parser.parse(onnx_model.SerializeToString())

        # 设置动态输入和输出
        min_shape = (1, 3, 224, 224)  # 最小输入形状
        max_shape = (1, 3, 512, 512)  # 最大输入形状
        opt_shape = (1, 3, 384, 384)  # 优选输入形状

        profile = builder.create_optimization_profile()
        profile.set_shape("input", min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        # 构建 TensorRT 引擎
        engine = builder.build_engine(network, config)

        # 保存引擎到文件
        # with open(engine_file_path, "wb") as f:
        #     f.write(engine.serialize())

        return engine


def infer(onnx_file_path, trt_file_path):
    # 加载类别索引和类别名称映射
    class_file = r"D:\workspace\data\dl\flower\flower.json"
    with open(class_file, "r", encoding="utf-8") as fr:
        cls2idx = json.load(fr)
    idx2cls = {v: k for k, v in cls2idx.items()}
    num_classes = 5
    batch_size = 8

    engine = get_engine2(onnx_file_path, trt_file_path)

    # 创建 TensorRT 执行上下文
    context = engine.create_execution_context()

    # 分配 GPU 内存
    d_input = cuda.mem_alloc(batch_size * 3 * 224 * 224 * 4)  # 3 是输入通道数，512x512 是输入形状，4 是 float32 的字节数
    d_output = cuda.mem_alloc(batch_size * num_classes * 4)  # num_classes 是输出通道数

    # 推理代码
    loader = batch_images_gen(image_dir=r"D:\workspace\data\dl\flower\test",
                              batch_size=batch_size,
                              img_w=224,
                              img_h=224,
                              mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225))
    len_dataset = 0
    start_time = time.time()
    for images, image_infos in loader:
        len_dataset += len(image_infos)
        input_data = images

        # context.set_binding_shape(0, (batch_size, 3, 224, 224))

        # 在 GPU 上分配内存并传输输入数据
        d_input = cuda.mem_alloc(input_data.nbytes)
        cuda.memcpy_htod(d_input, input_data)

        # 创建输出数据的 GPU 内存
        output = np.empty((batch_size, num_classes), dtype=np.float32)
        d_output = cuda.mem_alloc(output.nbytes)

        # 执行推理
        context.execute_v2(bindings=[int(d_input), int(d_output)])

        output_data = np.empty((batch_size, num_classes), dtype=np.float32)
        # 从 GPU 内存中获取输出
        cuda.memcpy_dtoh(output_data, d_output)

        preds = softmax(output_data)
        scores, indices = np.max(preds, axis=-1), np.argmax(preds, axis=-1)
        for ii, s, i in zip(image_infos, scores, indices):
            print(f"file_name: {ii.name} pred_class: {idx2cls[int(i)]} pred_score: {s}")
        break
    cost_time = time.time() - start_time
    print(f"inference time: {cost_time:.2f} s  fps: {1 / (cost_time / len_dataset):.2f}")

    # 清理资源
    del context
    del engine


def softmax(x):
    """ softmax function """

    # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x -= np.max(x, axis=1, keepdims=True)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


if __name__ == "__main__":
    onnx_file = r"D:\workspace\data\training_data\resnet50\pths\resnet50-0.934.onnx"
    engine_file = r"D:\workspace\data\training_data\resnet50\pths\resnet_engine2.trt"
    infer(onnx_file, engine_file)
    # get_engine(onnx_file, engine_file)

