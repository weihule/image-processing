import onnxruntime
import numpy as np
from typing import Union, Dict, List


class OnnxModel(object):
    def __init__(self, model_path, providers=None, warmup=True):
        """
        Args:
            model_path: onnx权重文件路径
            providers: ['CUDAExecutionProvider', 'CPUExecutionProvider',
                        'TensorrtExecutionProvider', 'TensorrtExecutionProvider']
            warmup: 是否进行warmup（首次推理通常较慢）
        """
        if providers is None:
            providers = ['CPUExecutionProvider']

        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {model_path}\n错误: {str(e)}")

        # 获取输入信息
        self.input_names = []
        self.input_shapes = []
        self.input_types = []
        self.input_dtypes = []
        self.dynamic_input_dims = {}  # 记录动态维度

        for inp in self.session.get_inputs():
            self.input_names.append(inp.name)
            self.input_shapes.append(inp.shape)
            self.input_types.append(inp.type)

            # 获取numpy数据类型, inp.type类似这样:  tensor(float)
            dtype_str = inp.type.split('(')[-1].rstrip(')')
            print(f"inp.name:{inp.name} inp.shape:{inp.shape} inp.type:{inp.type}")
            np_dtype = self._onnx_dtype_to_numpy(dtype_str)
            self.input_dtypes.append(np_dtype)

            # 检测动态维度
            # 如果是动态，inp.shape可能是: ['batch_size', 3, 'height', 'width'], 对应的dynamic_dims就是: {0: True, 2: True, 3: True}
            dynamic_dims = {}
            for i, dim in enumerate(inp.shape):
                if isinstance(dim, str) or dim is None:
                    dynamic_dims[i] = True
            if dynamic_dims:
                self.dynamic_input_dims[inp.name] = dynamic_dims

        # 获取输出信息
        self.output_names = []
        self.output_shapes = []
        self.output_types = []
        self.output_dtypes = []

        for out in self.session.get_outputs():
            self.output_names.append(out.name)
            self.output_shapes.append(out.shape)
            self.output_types.append(out.type)

            dtype_str = out.type.split('(')[-1].rstrip(')')
            np_dtype = self._onnx_dtype_to_numpy(dtype_str)
            self.output_dtypes.append(np_dtype)

        # 打印模型信息
        self._print_model_info(model_path)

        if warmup:
            self._warmup()

    @staticmethod
    def _onnx_dtype_to_numpy(dtype_str: str) -> np.dtype:
        """把onnx数据类型字符串转换成numpy数据类型"""
        dtype_map = {
            'float': np.float32,
            'double': np.float64,
            'int32': np.int32,
            'int64': np.int64,
            'int16': np.int16,
            'int8': np.int8,
            'uint32': np.uint32,
            'uint64': np.uint64,
            'uint16': np.uint16,
            'uint8': np.uint8,
            'bool': np.bool_,
        }
        base_type = dtype_str.lower().split('_')[0] if '_' in dtype_str else dtype_str.lower()
        return dtype_map.get(base_type, np.float32)

    def _print_model_info(self, model_path: str):
        """打印模型相信信息"""
        print(f"\n{'-' * 60}")
        print(f"✓ 模型加载成功: {model_path}")
        print(f"✓ 推理设备: {self.session.get_providers()}")
        print(f"---- 输入节点: {len(self.input_names)} ----")
        for name, shape, dtype, np_dtype in zip(
                self.input_names, self.input_shapes, self.input_types, self.input_dtypes
        ):
            dynamic_info = ""
            if name in self.dynamic_input_dims:
                dynamic_dims = list(self.dynamic_input_dims[name].keys())
                dynamic_info = f" [动态维度: {dynamic_dims}]"
            print(f"    - {name}")
            print(f"      Shape: {shape} {dynamic_info}")
            print(f"      Type: {dtype}")
            print(f"      NumPy dtype: {np_dtype}")

        print(f"---- 输出节点: {len(self.output_names)} ----")
        for name, shape, dtype, np_dtype in zip(
            self.output_names, self.output_shapes, self.output_types, self.output_dtypes
        ):
            print(f"    - {name}:")
            print(f"      Shape: {shape}")
            print(f"      Type: {dtype}")
            print(f"      NumPy dtype: {np_dtype}")
        print(f"{'-'*60}\n")

    def _warmup(self):
        """预热（首次推理会较慢）"""
        try:
            print(f"[Warmup] 正在进行模型预热...")
            dummy_inputs = {}
            for name, shape, np_dtype in zip(self.input_names, self.input_shapes, self.input_dtypes):
                # 替换动态维度1
                concrete_shape = []
                for i, s in enumerate(shape):
                    if i == 0 and (isinstance(s, str) or s is None):
                        concrete_shape.append(1)
                    elif i > 0 and (isinstance(s, str) or s is None):
                        concrete_shape.append(640)
                    else:
                        concrete_shape.append(s)
                concrete_shape = tuple(concrete_shape)
                dummy_inputs[name] = np.random.randn(*concrete_shape).astype(np_dtype)

                _ = self.session.run(self.output_names, dummy_inputs)
            print("[Warmup] 预热完成\n")
        except Exception as e:
            print(f"[Warmup] 警告: 预热失败 - {e}\n")

    def _prepare_inputs(self, inputs: Union[np.ndarray, Dict[str, np.ndarray]]):
        """把不同格式的输入转换为字典"""
        if isinstance(inputs, Dict):
            return inputs

        if isinstance(inputs, np.ndarray):
            if len(self.input_names) > 1:
                raise ValueError(f"模型有 {len(self.input_names)} 个输入节点: {self.input_names},"
                                 f"但只提供了一个numpy数组作为输入")
            else:
                return {self.input_names[0]: inputs}
        else:
            raise TypeError(f"不支持的输入类型: {type(inputs)}")

    def _validate_input_shape(self, input_name: str, actual_shape: tuple):
        """检查输入形状是否兼容"""
        expected_shape = self.input_shapes[self.input_names.index(input_name)]

        # 获取动态维度信息
        dynamic_shapes = self.dynamic_input_dims.get(input_name, {})

        if len(actual_shape) != len(expected_shape):
            raise ValueError(f"维度不匹配，期望: {expected_shape} 实际: {actual_shape}")

        # 检查每个维度
        for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
            if i in dynamic_shapes:
                # 动态维度，只检查实际值为正
                if actual <= 0:
                    raise ValueError(f"输入{input_name}的第{i}维(动态)的值必须为正")
            else:
                if expected != actual:
                    raise ValueError(
                        f"输入 '{input_name}' 第{i}维不匹配。"
                        f"期望: {expected}, 实际: {actual}"
                    )

    def _validate_and_convert_inputs(self,
                                     input_feed: Dict[str, np.ndarray],
                                     auto_convert_dtype=True
                                     ):
        """验证输入数据并做转换"""
        # 检查输入是否有遗漏
        missing_inputs = set(self.input_names) - set(input_feed.keys())
        if missing_inputs:
            raise ValueError(f"缺少以下输入节点: {missing_inputs}")

        # 检查是否有多余的
        extra_inputs = set(input_feed.keys()) - set(self.input_names)
        if missing_inputs:
            raise ValueError(f"⚠️ 警告: 提供了未知节点: {extra_inputs}")

        # 检查每个输入
        for name, data in input_feed.items():
            if not isinstance(data, np.ndarray):
                raise TypeError(f"输入 '{name}' 必须是 numpy.ndarray，当前是 {type(data)}")

            # 检查并转换数据类型
            expected_dtype = self.input_dtypes[self.input_names.index(name)]
            # print(f"---- {name} {data.shape} {data.dtype} {expected_dtype} {data.dtype == expected_dtype}")
            if data.dtype != expected_dtype:
                if auto_convert_dtype:
                    print(f"⚠️ 自动转换 '{name}' 的数据类型, 实际:{data.dtype} -> 期望:{expected_dtype}")
                    input_feed[name] = data.astype(expected_dtype)
                else:
                    raise TypeError(
                        f"输入 '{name}' 的数据类型不匹配。"
                        f"期望: {expected_dtype}, 实际: {data.dtype}\n"
                    )
            # 确保内存连续性
            if not data.flags.c_contiguous:
                print(f"⚠️ '{name}' 的内存不连续，正在转换...")
                input_feed[name] = np.ascontiguousarray(data)

            # 检查形状兼容性
            self._validate_input_shape(name, data.shape)

        return input_feed

    def _format_outputs(
        self,
        outputs: List[np.ndarray]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """格式化输出"""
        if len(outputs) == 1:
            return outputs[0]
        else:
            return {name: output for name, output in zip(self.output_names, outputs)}

    def __call__(self,
                 inputs: Union[np.ndarray, Dict[str, np.ndarray]],
                 auto_convert_type=True
                 ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        input_feed = self._prepare_inputs(inputs)
        input_feed = self._validate_and_convert_inputs(input_feed)

        # 执行推理
        try:
            outputs = self.session.run(self.output_names, input_feed)
        except Exception as e:
            raise RuntimeError(f"推理失败: {e}")

        return self._format_outputs(outputs)


def test():
    onnx_path = r'D:\workspace\weight_data\pre_weight\yolo11\yolo11l.onnx'
    om = OnnxModel(model_path=onnx_path)
    input1 = np.random.randn(1, 3, 640, 640)
    print(f"======== input1.dtype: {input1.dtype}")
    out1 = om(input1)


if __name__ == "__main__":
    test()

