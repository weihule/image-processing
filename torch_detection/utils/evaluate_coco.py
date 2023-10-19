import os
import time
from tqdm import tqdm
import json
import torch
from pycocotools.cocoeval import COCOeval
from .util import AverageMeter


def evaluate_coco_detection(test_loader, model, criterion, decoder, config):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    test_dataset = config.test_dataset
    ids = [idx for idx in range(len(test_dataset))]
    batch_size = config.batch_size

    with torch.no_grad():
        results, image_ids = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for i, data in tqdm(enumerate(test_loader)):
            images, annots,  = data['image'], data['annots']
            scales, sizes = data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()



