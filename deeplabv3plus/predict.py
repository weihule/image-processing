import argparse
from pathlib import Path
import yaml
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as TF

import network
from datasets import VOCSegmentation
from utils.utils import DictWrapper

DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def build_model(opts):
    model = network.modeling.__dict__[opts.model.name](
        num_classes=opts.model.num_classes,
        output_stride=opts.model.output_stride,
    )
    if getattr(opts.model, "separable_conv", False) and "plus" in opts.model.name:
        network.convert_to_separable_conv(model.classifier)
    return model


def load_weights(model, ckpt_path: Path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model_state", ckpt)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model


def preprocess(image: Image.Image, size, mean, std):
    orig_size = image.size  # (w, h)
    if size is not None:
        image = image.resize((size[1], size[0]), resample=Image.BILINEAR)
    img_tensor = TF.to_tensor(image)
    img_tensor = TF.normalize(img_tensor, mean=mean, std=std)
    return img_tensor.unsqueeze(0), orig_size, image.size


def postprocess(logits: torch.Tensor, orig_size, out_size):
    preds = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
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


def iter_images(path: Path):
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in path.glob("*") if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser(description="DeepLabV3+ single-image prediction")
    parser.add_argument("--config", type=str, default="./config.yaml", help="config path")
    parser.add_argument("--ckpt", type=str, default="./runs/best_ckpt.pth", help="checkpoint path")
    parser.add_argument("--image", type=str, required=True, help="image file or directory")
    parser.add_argument("--out-dir", type=str, default="./outputs", help="output directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--input-size", type=str, default=None, help="H,W or single int")
    parser.add_argument("--alpha", type=float, default=0.6, help="overlay alpha")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    opts = DictWrapper(config)

    if hasattr(opts, "dataset"):
        mean = getattr(opts.dataset, "mean", DEFAULT_MEAN)
        std = getattr(opts.dataset, "std", DEFAULT_STD)
    else:
        mean = DEFAULT_MEAN
        std = DEFAULT_STD

    input_size = parse_size(args.input_size)
    if input_size is None and hasattr(opts, "dataset") and hasattr(opts.dataset, "crop_size"):
        input_size = (int(opts.dataset.crop_size), int(opts.dataset.crop_size))

    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    model = build_model(opts)
    model = load_weights(model, Path(args.ckpt), device)
    model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(Path(args.image))
    if not image_paths:
        raise FileNotFoundError(f"No images found at {args.image}")

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        input_tensor, orig_size, resized_size = preprocess(image, input_size, mean, std)
        with torch.no_grad():
            logits = model(input_tensor.to(device))
        mask = postprocess(logits, orig_size, resized_size)
        label_path, overlay_path = save_outputs(mask, image, out_dir, img_path.stem, args.alpha)
        print(f"Saved: {label_path} and {overlay_path}")


if __name__ == "__main__":
    main()
