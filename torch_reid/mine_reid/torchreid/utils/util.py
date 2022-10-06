import os
import sys
import errno
import torch
from torch.utils import model_zoo

__all__ = [
    'Logger',
    'save_checkpoints',
    'init_pretrained_weights',
    'mkdir_if_missing'
]


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_checkpoints(states, isbest, save_dir, checkpoint_name):
    mkdir_if_missing(save_dir)
    torch.save(states, os.path.join(save_dir, checkpoint_name))
    if isbest:
        torch.save(states['model_state_dict'], os.path.join(save_dir, 'best_model.pth'))


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


def init_pretrained_weights(model, load_dir):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    # pretrain_dict = model_zoo.load_url(model_url, model_dir=load_dir)
    if load_dir is None:
        print('No pretrained weight')
        return
    pretrain_dict = torch.load(load_dir)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape}

    model_dict.update(pretrain_dict)
    print('load {} layers params , all {} layers'.format(len(pretrain_dict), len(model_dict)))
    model.load_state_dict(model_dict)


if __name__ == "__main__":
    sys.stdout = Logger(fpath='D:\\Desktop\\demo.log')
    print('this is a test')
    print('=========')
    print('8888888')

