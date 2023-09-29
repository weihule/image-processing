import os
import sys
import torch
import errno

__all__ = [
    'Logger',
    'mkdir_if_missing',
    'load_pretrained_weights',
    'save_checkpoints'
]


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


def load_pretrained_weights(model, weight_path=None):
    if weight_path is None:
        print('No pretrained weight')
        return
    pretrain_dict = torch.load(weight_path)
    model_dict = model.state_dict()

    valid_pretrain_dict = {
        k: v for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(valid_pretrain_dict)
    print('load {} layers params , all {} layers'.format(len(valid_pretrain_dict), len(model_dict)))
    model.load_state_dict(model_dict)


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


if __name__ == "__main__":
    process_checkpoints()

