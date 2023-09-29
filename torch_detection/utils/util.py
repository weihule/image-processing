import os
import sys
import logging
from logging import handlers
from thop import profile, clever_format
import errno
import torch
import torch.nn.functional as F


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


def compute_macs_and_params(input_image_size, model):
    assert isinstance(input_image_size, int) or isinstance(input_image_size, tuple)\
        or isinstance(input_image_size, list), "Illegal input_image_size type!"
    if isinstance(input_image_size, int):
        macs_input = torch.randn(1, 3, input_image_size,
                                 input_image_size).cpu()
    else:
        macs_input = torch.randn(1, 3, input_image_size[0],
                                 input_image_size[1]).cpu()

    model = model.cpu()
    macs, params = profile(model, inputs=(macs_input, ), verbose=False)
    macs, params = clever_format([macs, params], '%.3f')
    return macs, params


def custom_cross_entropy(input_data, target, num_class, use_custom=True):
    """
    :param use_custom: bool
    :param input_data: [N, num_class]
    :param target: [N]
    :param num_class: int
    :return:
    """
    if use_custom:
        one_hot = F.one_hot(target, num_classes=num_class).float()  # [N, num_class]
        custom_softmax = torch.exp(input_data) / torch.sum(torch.exp(input_data), dim=1).reshape((-1, 1))
        losses = -torch.sum(one_hot * torch.log(custom_softmax)) / input_data.shape[0]
    else:
        # 1
        # log_soft = F.log_softmax(input_data, dim=1)
        # losses = F.nll_loss(log_soft, target)

        # 2
        losses = F.cross_entropy(input_data, target)

    return losses


def custom_bce(input_data, target, num_class, use_custom=True):
    one_hot_target = F.one_hot(target, num_classes=num_class).float()

    if use_custom:
        print(input_data)
        print(one_hot_target)
        losses = -one_hot_target * torch.log(torch.sigmoid(input_data)) \
                 - (1 - one_hot_target) * (torch.log(1 - torch.sigmoid(input_data)))
        losses = losses.mean()
    else:
        # losses = F.binary_cross_entropy(input_data, one_hot_target)
        losses = F.binary_cross_entropy_with_logits(input_data, one_hot_target)

    return losses


if __name__ == "__main__":
    inputs = torch.rand((3, 4))
    labels = torch.tensor([0, 2, 3])
    custom_ce_loss = custom_cross_entropy(input_data=inputs,
                                          target=labels,
                                          num_class=4,
                                          use_custom=True)
    offical_ce_loss = custom_cross_entropy(input_data=inputs,
                                           target=labels,
                                           num_class=4,
                                           use_custom=False)

    custom_bce_loss = custom_bce(input_data=inputs,
                                 target=labels,
                                 num_class=4,
                                 use_custom=True)
    offical_bce_loss = custom_bce(input_data=inputs,
                                 target=labels,
                                 num_class=4,
                                 use_custom=False)
    print(custom_bce_loss, offical_bce_loss)

    