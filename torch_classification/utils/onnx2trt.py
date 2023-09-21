import os
import cv2

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
USE_FP16 = True
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
            config.max_workspace_size = 1 << 28     # 256MiB
            builder.max_batch_size = 1
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
            network.get_input(0).shape = [1, 3, 224, 224]
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


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def main():
    onnx_file = r"D:\workspace\data\training_data\resnet50\pths\resnet50-0.934.onnx"
    engine_file = r"D:\workspace\data\training_data\resnet50\pths\resnet_engine2.trt"
    # get_engine(onnx_file_path=onnx_file,
    #            engine_file_path=engine_file)
    image = process_image(img_path=r"D:\workspace\data\dl\flower\test\daisy03.jpg")
    with get_engine(onnx_file, engine_file) as engine, \
        engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        inputs[0].host = image
        trt_outputs = do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    print(trt_outputs)


def softmax(x):
    """ softmax function """

    # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x -= np.max(x, axis=1, keepdims=True)

    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    return x


if __name__ == "__main__":
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
    input_data = process_image(img_path=r"D:\workspace\data\dl\flower\test\roses02.jpg").astype(np.float32)

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
    print(scores, indices)

