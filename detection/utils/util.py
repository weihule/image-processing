import os
import sys
import numpy as np
import random
import time
from thop import profile, clever_format
import errno
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

__all__ = [
    "set_seed",
    "worker_seed_init_fn",
    "Logger",
    "mkdir_if_missing",
    "load_state_dict",
    "compute_macs_and_params",
    "AverageMeter"
]


def set_seed(seed):
    # for hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    # for python and numpy
    random.seed(seed)
    np.random.seed(seed)
    # for cpu gpu
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def worker_seed_init_fn(worker_id, num_workers, local_rank, seed):
    # worker_seed_init_fn function will be called at the beginning of each epoch
    # for each epoch the same worker has same seed value,so we add the current time to the seed
    worker_seed = num_workers * local_rank + worker_id + seed + int(
        time.time())
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'a', encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_checkpoints(states, save_state, isbest, save_dir, checkpoint_name):
    mkdir_if_missing(save_dir)
    if save_state:
        torch.save(states, os.path.join(save_dir, checkpoint_name))
    if isbest:
        torch.save(states['model_state_dict'], os.path.join(save_dir, 'best_model.pth'))


def process_checkpoints():
    weight_path = 'D:\\workspace\\data\\detection_data\\yolox\\yolox_m.pth'
    save_path = 'D:\\workspace\\data\\detection_data\\yolox\\yolox_m_model_weights.pth'
    checkpoints = torch.load(weight_path)
    model_weight = checkpoints['model']
    torch.save(model_weight, save_path)


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    if saved_model_path is None:
        print('No pretrained model file !')
        return
    save_state_dict = torch.load(saved_model_path, map_location='cpu')
    filtered_state_dict = {name: weight for name, weight in save_state_dict.items()
                           if name in model.state_dict() and weight.shape == model.state_dict()[name].shape
                           and not (name in excluded_layer_name)}
    if len(filtered_state_dict) == 0:
        print('have pre_weight, but not match model')
        return
    else:
        print('pretrained layers: {},  loading {} layers parameters !'.format(
            len(save_state_dict), len(filtered_state_dict)
        ))
        model.load_state_dict(filtered_state_dict, strict=False)
        return model


def compute_macs_and_params(input_image_size, model, device):
    assert isinstance(input_image_size, int) or isinstance(input_image_size, tuple)\
        or isinstance(input_image_size, list), "Illegal input_image_size type!"
    if isinstance(input_image_size, int):
        macs_input = torch.randn(1, 3, input_image_size,
                                 input_image_size).to(device)
    else:
        macs_input = torch.randn(1, 3, input_image_size[0],
                                 input_image_size[1]).to(device)

    model = model.to(device)
    macs, params = profile(model, inputs=(macs_input, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    return macs, params


class AverageMeter:
    """
    Compute and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    ave = AverageMeter()
    ave.update(15, 2)
    print(ave.val, ave.avg, ave.sum)

    ave.update(17, 1)
    print(ave.val, ave.avg, ave.sum)

    