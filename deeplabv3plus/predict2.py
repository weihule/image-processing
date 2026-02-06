import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from datasets import VOCSegmentation

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def parse_size(size_str):
    if size_str is None:
        return None
    s = str(size_str)
    if "," in s:
        parts = s.split(",")
        if len(parts) != 2:
            raise ValueError("input size must be like 'H,W' or a single int")
        h, w = [int(p) for p in parts]
    else:
        h = w = int(s)
    return (h, w)


def preprocess(image: Image.Image, size, mean, std, layout):
    orig_size = image.size  # (w, h)
    if size is not None:
        image = image.resize((size[1], size[0]), resample=Image.BILINEAR)
    img = np.array(image).astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean_arr) / std_arr

    if layout == "nchw":
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
    elif layout == "nhwc":
        img = np.expand_dims(img, axis=0)
    else:
        raise ValueError("layout must be nchw or nhwc")

    return img.astype(np.float32), orig_size, image.size


def postprocess(logits, orig_size, out_size, output_layout):
    if output_layout == "nhwc":
        logits = np.transpose(logits, (0, 3, 1, 2))
    elif output_layout != "nchw":
        raise ValueError("output layout must be nchw or nhwc")

    preds = np.argmax(logits, axis=1)[0].astype(np.uint8)
    if out_size != orig_size:
        preds = np.array(Image.fromarray(preds).resize(orig_size, resample=Image.NEAREST))
    return preds


def save_outputs(mask, orig_image: Image.Image, out_dir: Path, stem: str, alpha: float):
    color = VOCSegmentation.decode_target(mask)
    label_img = Image.fromarray(color.astype(np.uint8))
    label_path = out_dir / f"{stem}_label.png"
    label_img.save(label_path)

    if orig_image.size != label_img.size:
        orig_image = orig_image.resize(label_img.size, resample=Image.BILINEAR)

    orig_np = np.array(orig_image).astype(np.float32)
    color_np = color.astype(np.float32)
    overlay = (orig_np * (1.0 - alpha) + color_np * alpha).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)
    overlay_path = out_dir / f"{stem}_overlay.png"
    overlay_img.save(overlay_path)

    return label_path, overlay_path


def infer_onnx(model_path: Path, input_array: np.ndarray, device: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is not installed") from exc

    if device.startswith("cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_array})
    return outputs[0]


def infer_tensorrt(engine_path: Path, input_array: np.ndarray):
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("TensorRT/pycuda is not installed") from exc

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    input_binding = None
    output_binding = None
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            input_binding = i
        else:
            output_binding = i

    if input_binding is None or output_binding is None:
        raise RuntimeError("Could not find input/output bindings in TensorRT engine")

    input_shape = tuple(input_array.shape)
    if engine.is_shape_binding(input_binding) or context.get_binding_shape(input_binding)[0] == -1:
        context.set_binding_shape(input_binding, input_shape)

    output_shape = tuple(context.get_binding_shape(output_binding))
    if any(dim < 0 for dim in output_shape):
        raise RuntimeError("Dynamic output shape is not supported in this simple runner")

    input_size = int(np.prod(input_shape)) * np.dtype(np.float32).itemsize
    output_size = int(np.prod(output_shape)) * np.dtype(np.float32).itemsize

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    bindings = [0] * engine.num_bindings
    bindings[input_binding] = int(d_input)
    bindings[output_binding] = int(d_output)

    stream = cuda.Stream()
    cuda.memcpy_htod_async(d_input, input_array, stream)
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    output = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

    return output


def infer_rknn(model_path: Path, input_array: np.ndarray):
    try:
        from rknnlite.api import RKNNLite
        rknn = RKNNLite()
        ret = rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNNLite load failed: {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNNLite runtime init failed: {ret}")
        outputs = rknn.inference(inputs=[input_array])
        rknn.release()
        return outputs[0]
    except ImportError:
        try:
            from rknn.api import RKNN
        except ImportError as exc:
            raise RuntimeError("RKNN toolkit is not installed") from exc
        rknn = RKNN()
        ret = rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load failed: {ret}")
        ret = rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN runtime init failed: {ret}")
        outputs = rknn.inference(inputs=[input_array])
        rknn.release()
        return outputs[0]


def iter_images(path: Path):
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in path.glob("*") if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ non-PyTorch prediction")
    parser.add_argument("--backend", type=str, required=True, choices=["onnx", "tensorrt", "rknn"], help="inference backend")
    parser.add_argument("--model", type=str, required=True, help="model file path")
    parser.add_argument("--image", type=str, required=True, help="image file or directory")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu (onnx only)")
    parser.add_argument("--input-size", type=str, default=None, help="H,W or single int")
    parser.add_argument("--layout", type=str, default="nchw", choices=["nchw", "nhwc"], help="input layout")
    parser.add_argument("--output-layout", type=str, default="nchw", choices=["nchw", "nhwc"], help="output layout")
    parser.add_argument("--alpha", type=float, default=0.6, help="overlay alpha")
    args = parser.parse_args()

    input_size = parse_size(args.input_size)
    mean = DEFAULT_MEAN
    std = DEFAULT_STD

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(Path(args.image))
    if not image_paths:
        raise FileNotFoundError(f"No images found at {args.image}")

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        input_array, orig_size, resized_size = preprocess(image, input_size, mean, std, args.layout)

        if args.backend == "onnx":
            logits = infer_onnx(Path(args.model), input_array, args.device)
        elif args.backend == "tensorrt":
            logits = infer_tensorrt(Path(args.model), input_array)
        else:
            logits = infer_rknn(Path(args.model), input_array)

        mask = postprocess(logits, orig_size, resized_size, args.output_layout)
        label_path, overlay_path = save_outputs(mask, image, out_dir, img_path.stem, args.alpha)
        print(f"Saved: {label_path} and {overlay_path}")


if __name__ == "__main__":
    main()
