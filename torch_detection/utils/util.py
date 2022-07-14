import os
import sys
import logging
from logging import handlers
import torch


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
        model.load_state_dict(filtered_state_dict, strict=False)

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('this is a info')
