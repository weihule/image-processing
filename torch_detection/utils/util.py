import os
import sys
import logging
from logging import handlers
import torch
import torch.nn.functional as F


def get_logger(name, log_dir):
    """
    Args:
        name(str): name of logger
        log_dir(str): path of log
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = handlers.TimedRotatingFileHandler(filename=info_name,
                                                     when='D',
                                                     encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger


def load_state_dict(saved_model_path, model, excluded_layer_name=()):
    if not saved_model_path:
        print('No pretrained model file!')
        return
    saved_state_dict = torch.load(saved_model_path,
                                  map_location=torch.device('cpu'))

    filtered_state_dict = {
        name: weight
        for name, weight in saved_state_dict.items()
        if name in model.state_dict() and not any(excluded_name in name for excluded_name in excluded_layer_name)
        and weight.shape == model.state_dict()[name].shape
    }

    if len(filtered_state_dict) == 0:
        print('No pretrained parameters to load!')
    else:
        print(f'loading {len(filtered_state_dict)} layers parameters')
        model.load_state_dict(filtered_state_dict, strict=False)

    return


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
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s %(levelname)s %(message)s')
    # logging.info('this is a info')

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
    
    # print(custom_ce_loss, offical_ce_loss)
    # inputs = torch.tensor([[0.9893, 0.4266, 0.7232, 0.5329],
    #                       [0.8937, 0.9865, 0.8116, 0.4274],
    #                       [0.6525, 0.1494, 0.6610, 0.6263]])

    # inputs = torch.tensor([[0.9893, 0.1023, 0.0987, 0.0777],
    #                       [0.1023, 0.0987, 0.9654, 0.1098],
    #                       [0.9999, 0.0078, 0.0976, 0.9999]])

    custom_bce_loss = custom_bce(input_data=inputs,
                                 target=labels,
                                 num_class=4,
                                 use_custom=True)
    offical_bce_loss = custom_bce(input_data=inputs,
                                 target=labels,
                                 num_class=4,
                                 use_custom=False)
    print(custom_bce_loss, offical_bce_loss)

    