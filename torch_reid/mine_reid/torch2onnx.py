import torch
import onnx
import onnxsim
import onnxruntime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchreid import models


def convert_torch2onnx_model(model, inputs, save_file_path, opset_version=11, use_onnxsim=True):
    print(f'starting export with onnx version {onnx.__version__}...')
    dynamic_axes = {
        'input': {0: 'batch_size'},  # 这么写表示第0维可以变化
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


if __name__ == "__main__":
    pth_path = r'D:\Desktop\best_model.pth'
    net_name = 'osnet_x1_0_origin'
    net = models.init_model(name=net_name,
                            num_classes=751)
    for idx, m in enumerate(net.modules()):
        print(idx, type(m))
    net.eval()
    # model.load_state_dict(torch.load(pth_path))

    # input_data = torch.randn(1, 3, 256, 128)
    # save_path = "D:\\Desktop\\" + net_name + ".onnx"
    # convert_torch2onnx_model(model=net,
    #                          inputs=input_data,
    #                          save_file_path=save_path,
    #                          use_onnxsim=False)
