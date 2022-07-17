import os
import sys
import glob
import cv2
import numpy as np
import torch
from torch.backends import cudnn

base_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from torch_detection.retinanet.retina_decode import RetinaNetDecoder
from network_files.retinanet_model import resnet50_retinanet
from config import Config


class InferResizer:
    def __init__(self, resize=Config.input_image_size):
        self.resize = resize

    def __call__(self, image):
        h, w, c = image.shape
        if h >= w:
            scale = self.resize / h
            resize_w = int(round(scale * w))
            resize_h = self.resize
        else:
            scale = self.resize / w
            resize_w = self.resize
            resize_h = int(round(scale * h))

        resize_img = cv2.resize(image, (resize_w, resize_h))
        padded_img = np.zeros((self.resize, self.resize, 3), dtype=np.float32)
        padded_img[:resize_h, :resize_w, :] = resize_img
        padded_img = torch.from_numpy(padded_img)

        return padded_img


def main():
    img_root = '/workshop/weihule/data/detection_data/test_images/*.jpg'
    model_path = '/workshop/weihule/data/detection_data/retinanet/checkpoints/resnet50_retinanet-metric79.783.pth'

    infer_batch = 4
    cudnn.benchmark = True
    cudnn.deterministic = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50_retinanet(num_classes=80).to(device)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    checkpoint = {key.replace('P', 'p'): value for key, value in checkpoint.items()}

    model.load_state_dict(checkpoint)
    model.eval()
    decoder = RetinaNetDecoder(image_w=Config.input_image_size,
                               image_h=Config.input_image_size)
    infer_resizer = InferResizer()
    img_lists = glob.glob(img_root)
    img_spilt_lists = [img_lists[start: start + infer_batch] for start in range(0, len(img_lists), infer_batch)]
    for img_spilt_list in img_spilt_lists:
        images_src = [cv2.imread(p) for p in img_spilt_list]
        images_name = [p.split(os.sep)[-1] for p in img_spilt_list]
        images = [infer_resizer(p).to(device) for p in images_src]
        images_tensor = torch.stack(images, dim=0)
        images_tensor = images_tensor.permute(0, 3, 1, 2).contiguous()
        cls_heads, reg_heads, batch_anchors = model(images_tensor)
        batch_scores, batch_classes, batch_pred_bboxes = decoder(cls_heads, reg_heads, batch_anchors)
        batch_scores, batch_classes, batch_pred_bboxes = \
            batch_scores.cpu().numpy(), batch_classes.cpu().numpy(), batch_pred_bboxes.cpu().numpy()
        for scores, classes, pred_bboxes, img, img_name in \
                zip(batch_scores, batch_classes, batch_pred_bboxes, images_src, images_name):
            for bbox in pred_bboxes:
                pt1 = np.asarray([bbox[0], bbox[1]], dtype=np.int32)
                pt2 = np.asarray([bbox[2], bbox[3]], dtype=np.int32)
                cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            cv2.imwrite(img_name, img)
        # break


if __name__ == "__main__":
    main()
