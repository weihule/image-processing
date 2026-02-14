import time
import warnings
import numpy as np
from packaging.version import Version
from typing import Dict, Union, Tuple, Optional
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.special import expected


class TrtModel(object):
    def __init__(self,
                 model_path: str,
                 gpu_id: int = 0,
                 verbose: bool = True
                 ):
        self._initialized = False
        self.verbose = verbose
        self.gpu_id = gpu_id
        self.cuda_ctx = None  # 手动管理CUDA上下文

        try:
            cuda.init()
            self.cuda_device = cuda.Device(gpu_id)
            self.cuda_ctx = self.cuda_device.make_context()
            if self.verbose:
                print(f"✓ CUDA context created on GPU {gpu_id}")

            self._check_version()

            # 创建Logger
            self.logger = trt.Logger(trt.Logger.WARNING)

            # 加载Engine
            with open(model_path, 'rb') as frb:
                engine_data = frb.read()
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
            if self.engine is None:
                raise RuntimeError("反序列化CUDA引擎失败")

            # 创建执行上下文
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(f"创建执行上下文失败")

            # 分析输入输出信息
            self._analyze_bindings()

            # 检测动态维度（移除未使用的dynamic_dims）
            self._check_dynamic_shapes()

            # 根据是否动态决定分配内存策略
            if self.has_dynamic_shapes:
                if not self.use_new_api:
                    raise RuntimeError(f"TensorRT >= 8.5才支持动态维度")

                self.device_buffers = {}
                self.host_outputs = {}
                self.current_shapes = {}
                self.current_alloc_sizes = {}  # 记录当前分配的字节大小（优化重分配判断）
            else:
                self._allocate_buffers_static()

            # 创建cuda流
            self.stream = cuda.Stream()

            # 完成初始化
            self._initialized = True
            self._print_model_info(model_path)

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"初始化失败{e}")

    def _check_version(self):
        """检查TensorRT版本"""
        trt_version = trt.__version__
        self.use_new_api = Version(trt_version) >= Version('8.5')

        # 双重验证
        if self.use_new_api:
            self.use_new_api = hasattr(trt.ICudaEngine, 'get_tensor_name')

        print(f"📋 TensorRT 版本: {trt_version}")

        # 打印cuda信息
        try:
            cuda_device = cuda.Device(0)
            device_name = cuda_device.name()
            device_name = (device_name.decode('utf-8', errors='ignore')
                           if isinstance(device_name, bytes) else device_name)
            device_cc = cuda_device.compute_capability()
            print(f"  CUDA Device: {device_name}")
            print(f"  Compute Capability: {device_cc}")

        except Exception as e:
            print(f"  CUDA Info: {e}")

    def _get_binding_info(self, idx: int) -> dict:
        """
        获取binding的详细信息
            以yolo11s为例，num_bindings是4，得到的info分别是
            {'name': 'input', 'shape': (1, 3, 960, 960), 'dtype': <DataType.FLOAT: 0>, 'is_input': True}
            {'name': 'onnx::Reshape_979', 'shape': (1, 144, 120, 120), 'dtype': <DataType.FLOAT: 0>, 'is_input': False}
            {'name': 'onnx::Reshape_1006', 'shape': (1, 144, 60, 60), 'dtype': <DataType.FLOAT: 0>, 'is_input': False}
            {'name': 'onnx::Reshape_1033', 'shape': (1, 144, 30, 30), 'dtype': <DataType.FLOAT: 0>, 'is_input': False}
            {'name': 'output', 'shape': (1, 84, 18900), 'dtype': <DataType.FLOAT: 0>, 'is_input': False}
        """
        info = {}
        if self.use_new_api:
            name = self.engine.get_tensor_name(idx)
            info['name'] = name
            info['shape'] = tuple(self.engine.get_tensor_shape(idx))
            info['dtype'] = self.engine.get_tensor_dtype(idx)
            info['is_input'] = self.engine.get_tensor_mode(idx) == trt.TensorIOMode.INPUT
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                info['name'] = self.engine.get_binding_name(idx)
                info['shape'] = tuple(self.engine.get_binding_shape(idx))
                info['dtype'] = self.engine.get_binding_dtype(idx)
                info['is_input'] = self.engine.binding_is_input(idx)
        return info

    def _analyze_bindings(self):
        """分析所有输入输出的绑定信息"""
        self.inputs = {}
        self.outputs = {}
        self.bindings_info = {}

        # num_tensors是一个整数，engine有几个输入和输出，num_bindings就是多少（想要可视化可以通过onnx格式来看）
        num_bindings = self.engine.num_io_tensors if self.use_new_api else self.engine.num_bindings

        for i in range(num_bindings):
            binding_info = self._get_binding_info(i)
            name = binding_info['name']
            if binding_info['is_input']:
                self.inputs[name] = binding_info
            else:
                self.outputs[name] = binding_info
            self.bindings_info[name] = binding_info
            self.bindings_info[name]['index'] = i       # 给每个binding_info添加index索引信息

        if not self.use_new_api:
            self.bindings = [None] * num_bindings

        print(f"发现 {len(self.inputs)} 个输入, {len(self.outputs)} 个输出")

    def _check_dynamic_shapes(self):
        """检测动态维度（移除未使用的self.dynamic_dims）"""
        self.has_dynamic_shapes = False
        self.dynamic_dims = {}  # 记录哪些tensor的哪些维度是动态的

        for name, info in self.bindings_info.items():
            dynamic_axes = []
            for idx, dim in enumerate(['shape']):
                if dim == -1 or dim == 0:
                    dynamic_axes.append(idx)
                    self.has_dynamic_shapes = True
            if dynamic_axes:
                self.dynamic_dims[name] = dynamic_axes
                io_type = "Input" if info['is_input'] else "Output"
                print(f"  ⚠️  {io_type:6} '{name}'有动态维度: {dynamic_axes} 动态形状:{info['shape']}")

        if not self.dynamic_dims:
            print("没有动态维度")

    @staticmethod
    def _trt_dtype_to_numpy(trt_dtype):
        """TensorRT数据类型转Numpy数据类型"""
        dtype_map = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF: np.float16,
            trt.DataType.INT8: np.int8,
            trt.DataType.INT32: np.int32,
            trt.DataType.BOOL: np.bool_,
        }
        if hasattr(trt.DataType, 'INT64'):
            dtype_map[trt.DataType.INT64] = np.int64
        if hasattr(trt.DataType, 'FLOAT64'):
            dtype_map[trt.DataType.FLOAT64] = np.float64
        return dtype_map.get(trt_dtype, np.float32)

    def _allocate_buffers_static(self):
        """分配GPU和CPU内存（仅用于静态shape）
        """
        print("\n💾 分配静态内存...")
        self.device_buffers = {}
        self.host_outputs = {}      # 只为输出分配host内存

        total_gpu_memory = 0
        total_cpu_memory = 0

        for name, info in self.bindings_info.items():
            np_dtype = self._trt_dtype_to_numpy(info['type'])
            dtype_size = np.dtype(np_dtype).itemsize

            # 计算buffer大小
            buffer_size = int(np.prod(info['shape'])) * dtype_size

            try:
                # 分配GPU内存
                d_buffer = cuda.mem_alloc(buffer_size)
                self.device_buffers[name] = d_buffer
                total_gpu_memory += d_buffer

                # 对于旧API，设置bindings
                if not self.use_new_api:
                    self.bindings[info['index']] = int(d_buffer)

                # 只为输出分配页锁定主机内存（输入直接从numpy传输）
                if not info['is_input']:
                    self.host_outputs[name] = cuda.pagelocked_empty(info['shape'],
                                                                    dtype=np_dtype)
                    total_cpu_memory += buffer_size

                io_type = "Input" if info['is_input'] else "Output"
                print(f"{io_type:6} {name:30} {str(info['shape']):30} {buffer_size/1024/1024:8.2f} MB")

            except Exception as e:
                raise RuntimeError(f"{name} 分配缓冲区失败: {e}")

        print(f"total_gpu_memory: {total_gpu_memory}")
        print(f"total_cpu_memory: {total_cpu_memory}")

    def _allocate_buffers_dynamic(self, actual_shapes: Dict[str, Tuple]):
        """根据实际shape动态分配内存(有缓存功能，只在必要时重新分配)"""
        for name, shape in actual_shapes.items():
            info = self.bindings_info[name]
            np_dtype = self._trt_dtype_to_numpy(info['dtype'])
            dtype_size = np.dtype(np_dtype).itemsize

            new_size = int(np.prod(shape)) * dtype_size

            # 检查是否需要重新分配
            need_realloc = False
            if name not in self.device_buffers:
                need_realloc = True
            elif name in self.current_shapes:
                old_size = int(np.prod(self.current_shapes[name])) * dtype_size
                # 只在size变大时或变小超过50%时重新分配（减少频繁重新分配）
                if new_size > old_size or new_size < old_size*0.5:
                    need_realloc = True
            else:
                need_realloc = True

            if need_realloc:
                # 释放旧buffer
                if name in self.device_buffers and self.device_buffers[name] is not None:
                    try:
                        self.device_buffers[name].free()
                    except Exception as e:
                        print(f"{name} 释放缓存失败: {e}")

                # 重新分配buffer（预留20 % 空间或至少1MB，减少频繁重新分配）
                alloc_size = max(int(new_size*1.2), new_size+1024*1024)

                try:
                    self.device_buffers[name] = cuda.mem_alloc(alloc_size)
                    self.current_shapes[name] = shape
                    io_type = "Input" if info['is_input'] else "Output"
                    print(f"{io_type:6} {name:30} {str(info['shape']):30} {alloc_size/1024/1024:8.2f} MB")
                except cuda.Error as e:
                    raise RuntimeError(f"{name} 分配GPU显存失败: {e}")

            # 为输出分配页锁定内存
            if not info['is_input']:
                if name not in self.host_outputs or self.host_outputs[name].shape != shape:
                    try:
                        self.host_outputs[name] = cuda.pagelocked_empty(shape, dtype=np_dtype)
                    except cuda.Error as e:
                        raise RuntimeError(f"{name} 分配页锁定内存失败: {e}")

    def _print_model_info(self, model_path: str):
        """打印模型相信信息"""
        print(f"\n{'-' * 60}")
        print(f"✓ 模型加载成功: {model_path}")
        print(f"---- 输入节点: {len(self.inputs)} ----")
        for name, info in self.inputs.items():
            np_dtype = info['dtype']
            trt_dtype = self._trt_dtype_to_numpy(np_dtype)
            print(f"    - {name}")
            print(f"      Shape: {info['shape']}")
            print(f"      Type: {trt_dtype}")
            print(f"      NumPy dtype: {np_dtype}")

        print(f"---- 输出节点: {len(self.outputs)} ----")
        for name, info in self.outputs.items():
            np_dtype = info['dtype']
            trt_dtype = self._trt_dtype_to_numpy(np_dtype)
            print(f"    - {name}:")
            print(f"      Shape: {info['shape']}")
            print(f"      Type: {trt_dtype}")
            print(f"      NumPy dtype: {np_dtype}")
        print(f"{'-'*60}\n")

    def _call_static(self, input_dict: Dict[str, np.ndarray]) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """静态shape推理"""
        try:
            for name, data in input_dict.items():
                if name not in self.inputs:
                    raise ValueError(f"未知输入名: {name}")
                expected_shape = self.inputs[name]['shape']
                expected_dtype = self._trt_dtype_to_numpy(self.inputs[name]['dtype'])

                # 检查并调整shape
                if data.shape != expected_shape:
                    if np.prod(data.shape) == np.prod(expected_shape):
                        data = data.reshape(expected_shape)
                    else:
                        raise ValueError(
                            f"{name} 输入形状不匹配, 输入:{data.shape}, 期望:{expected_shape}"
                        )
        except Exception as e:
            print(f"静态推理时报错: {e}")

    def cleanup(self):
        """清理资源"""
        if not hasattr(self, '__initialized'):
            return

        try:
            pass
        except Exception as e:
            print(f"清理资源失败 {e}")

def test():
    trt_path = r'D:\workspace\weight_data\pre_weight\yolo11\yolo11l.engine'
    tm = TrtModel(model_path=trt_path)


if __name__ == "__main__":
    test()




