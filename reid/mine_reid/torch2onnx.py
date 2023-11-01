import torch
import onnx
import warnings
import onnxsim
import onnxruntime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format
from torchreid import models

warnings.filterwarnings("ignore")


def convert_torch2onnx_model(model, inputs, save_file_path, opset_version=12, use_onnxsim=True):
    print(f'starting export with onnx version {onnx.__version__}...')
    dynamic_axes = {
        'input': {0: 'batch_size', 2: "height", 3: "width"},  # 这么写表示第0维可以变化
        'output': {0: 'batch_size'},
    }
    torch.onnx.export(model,
                      inputs,
                      save_file_path,
                      export_params=True,
                      verbose=False,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=opset_version,
                      do_constant_folding=True,
                      dynamic_axes=dynamic_axes)
    # load and check onnx model
    onnx_model = onnx.load(save_file_path)
    onnx.checker.check_model(onnx_model)

    # Simplify onnx model
    if use_onnxsim:
        print(f'using onnx-simplifier version {onnxsim.__version__}')
        onnx_model, check = onnxsim.simplify(
            onnx_model,
            dynamic_input_shape=False,
            input_shapes={'inputs': inputs.shape})
        assert check, 'assert onnxsim model check failed'
        onnx.save(onnx_model, save_file_path)
        print(f'onnxsim model is checked, convert onnxsim model success, saved as {save_file_path}')

    print(f"save finished: {save_file_path}")


def main(name, act_func, attention, aligned):
    pth_path = r'D:\Desktop\tempfile\weights\market1501\osnet_x1_0_9171_7915.pth'
    net = models.init_model(name=name,
                            num_classes=751,
                            act_func=act_func,
                            attention=attention,
                            loss='softmax_trip',
                            aligned=aligned)
    net.load_state_dict(torch.load(pth_path))
    net.eval()

    input_data = torch.randn(1, 3, 256, 128)
    save_path = r"D:\Desktop\tempfile\weights\market1501\osnet_x1_0_9171_7915.onnx"
    convert_torch2onnx_model(model=net,
                             inputs=input_data,
                             save_file_path=save_path,
                             use_onnxsim=False)


def test_model(name, act_func, attention, aligned):
    """
    测试模型的FLOPs、params以及是否可以转成ONNX
    """
    model = models.init_model(name=name,
                              num_classes=751,
                              act_func=act_func,
                              attention=attention,
                              loss='softmax_trip',
                              aligned=aligned)
    input_data = torch.randn(4, 3, 256, 128)
    outs, features, local_features = model(input_data)
    print('model output => ', outs.shape, features.shape, local_features.shape)
    Macs, params = profile(model, inputs=(torch.randn(1, 3, 256, 128),))
    Flops = Macs * 2
    Flops, params = clever_format([Flops, params], "%.3f")
    print(f"model: {name} \nattention: {attention} Flops: {Flops}, params: {params}")

    # save_path = "D:\\Desktop\\osnet.pth"
    # print(len(model.state_dict().keys()))
    # torch.save(model.state_dict(), save_path)


def test_onnx():
    """
    测试生成的onnx是否可以正常输出
    """
    onnx_file = "D:\\Desktop\\osnet.onnx"
    onnx_session = onnxruntime.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
    input_name = [onnx_session.get_inputs()[0].name]
    output_name = [onnx_session.get_outputs()[0].name]
    input_feed = {}
    for name in input_name:
        input_feed[name] = np.random.random(size=(4, 3, 256, 128)).astype(np.float32)
    outs = onnx_session.run(output_name, input_feed=input_feed)
    features = outs[0]
    print(type(features), features.shape)


if __name__ == "__main__":
    # osnet_x1_0_origin resnet50 sc_osnet_x1_0_origin
    model_name = "osnet_x1_0_origin"
    activation_function = "relu"
    attention_function = None
    # test_model(model_name, activation_function, attention_function, True)
    main(model_name, activation_function, attention_function, False)
    # test_onnx()
    # 2_ueAPYIhFgpOZdgFzhiKz_0hh0H7pBVGnOPulUU8PJxryMS

