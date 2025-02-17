import os
import sys
import onnx
import onnxsim
import onnxruntime
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from backbones.model_manager import init_model


def convert_torch2onnx_model(model, inputs, save_file_path, opset_version=14, use_onnxsim=True):
    print(f'starting export with onnx version {onnx.__version__}...')
    # dynamic_axes = {
    #     'input': {0: 'batch_size', 2: "height", 3: "width"},  # 第0、2、3维可以变化
    #     'output': {0: 'batch_size'},
    # }
    dynamic_axes = {
        'input': {2: "height", 3: "width"},  # 第2、3维可以变化
    }
    torch.onnx.export(model,
                      inputs,
                      save_file_path,
                      export_params=True,
                      verbose=True,
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=opset_version,
                      do_constant_folding=True,
                      dynamic_axes=dynamic_axes)
    # load and check onnx model
    onnx_model = onnx.load(save_file_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid.")

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


def main():
    root = r"D:\workspace\data\training_data"
    # model_name = "mobilenetv2_x1_0"
    # pth_name = "mobilenetv2_x1_0-0.9504.pth"

    model_name = "resnet50"
    pth_name = "resnet50-0.9421.pth"
    # pth_path = r"D:\workspace\data\training_data\resnet50\pths\resnet50-0.9421.pth"
    pth_path = os.path.join(root, model_name, "pths", pth_name)
    model = init_model(backbone_type=model_name,
                       num_classes=5)
    model.load_state_dict(torch.load(pth_path), strict=True)
    model.eval()
    input_data = torch.randn(1, 3, 224, 224)
    save_path = pth_path[:-3] + "onnx"

    convert_torch2onnx_model(model,
                             inputs=input_data,
                             save_file_path=save_path,
                             opset_version=12,
                             use_onnxsim=False)


if __name__ == "__main__":
    main()


