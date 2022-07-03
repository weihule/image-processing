import os
import sys
import logging
from logging import handlers


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.info('this is a info')
