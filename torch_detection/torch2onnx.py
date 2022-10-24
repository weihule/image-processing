import numpy as np
import torch
import onnxruntime
import onnxsim
import onnx
import warnings

warnings.filterwarnings('ignore')


def convert_torch_model_to_onnx_model(model,
                                      inputs,
                                      save_file_path,
                                      opset_version=13,
                                      use_onnxsim=True):
    print(f'Starting export with onnx version  {onnx.__version__}...')
    input_name = 'input'
    output_name0 = 'output0'
    output_name1 = 'output1'
    output_name2 = 'output2'
    output_name3 = 'output3'
    output_name4 = 'output4'
    output_name5 = 'output5'
    output_name6 = 'output6'
    output_name7 = 'output7'
    output_name8 = 'output8'
    output_name9 = 'output9'
    outputs = [output_name0, output_name1, output_name2, output_name3, output_name4,
               output_name5, output_name6, output_name7, output_name8, output_name9]
    torch.onnx.export(model=model,
                      args=inputs,
                      f=save_file_path,
                      export_params=True,
                      verbose=False,
                      input_names=[input_name],
                      output_names=outputs,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      dynamic_axes={
                          input_name: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name0: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name1: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name2: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name3: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name4: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name5: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name6: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name7: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name8: {0: 'batch_size', 2: 'in_height', 3: 'int_width'},
                          output_name9: {0: 'batch_size', 2: 'in_height', 3: 'int_width'}}
                      )
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


if __name__ == '__main__':
    arch = 'resnet50_retinanet'
    num_classes = 80
    train_model_path = 'D:\\Desktop\\tempfile\\best_model.pth'

    import models

    model = models.init_model(name=arch,
                              num_classes=num_classes,
                              pre_train_load_dir=None)
    checkpoint = torch.load(train_model_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)
    model.eval()

    images = torch.randn(1, 3, 640, 640)
    outs = model(images)
    print('1111', len(outs))

    save_file_path = 'D:\\Desktop\\tempfile\\best_model.onnx'
    convert_torch_model_to_onnx_model(model=model,
                                      inputs=images,
                                      save_file_path=save_file_path,
                                      use_onnxsim=False)

    # test_onnx_images = np.random.randn(1, 3, 400, 400).astype(np.float32)
    # model = onnx.load(save_file_path)
    # onnxruntime_session = onnxruntime.InferenceSession(save_file_path)
    # outputs = onnxruntime_session.run(None, dict(inputs=test_onnx_images))
    # print('1111,onnx result:', len(outputs))
