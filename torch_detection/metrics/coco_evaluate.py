import os
import sys
import collections
import numpy as np
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from pycocotools.cocoeval import COCOeval

sys.path.append(os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__))))
from util.avgmeter import AverageMeter


def evaluate_coco_detection(test_loader, model, criterion, decoder, args):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    test_dataset = test_loader.dataset
    ids = [i for i in range(len(test_dataset))]
    batch_size = int(test_loader.batch_size)

    with torch.no_grad():
        results, image_ids = [], []
        model_on_cuda = next(model.parameters()).is_cuda
        end = time.time()
        for i, data in enumerate(tqdm(test_loader)):
            images, annots, scales, sizes = data['img'], data['annot'], data['scale'], data['size']
            if model_on_cuda:
                images, annots = images.cuda(), annots.cuda()

            per_batch_ids = ids[i * batch_size: (i+1) * batch_size]

            torch.cuda.synchronize()
            data_time.update(time.time() - end, images.size(0))
            end = time.time()

            outs_tuple = model(images)
            loss_value = criterion(outs_tuple, annots)
            loss = sum(loss_value.values())
            losses.update(loss, images.shape[0])

            # scores shape:[batch_size,max_object_num]
            # classes shape:[batch_size,max_object_num]
            # bboxes shape[batch_size,max_object_num,4]
            scores, classes, boxes = decoder(outs_tuple)

            boxes /= np.expand_dims(np.expand_dims(scales, axis=-1), axis=-1)
            torch.cuda.synchronize()
            batch_time.update(time.time() - end, images.shape[0])

            for per_image_scores, per_image_classes, per_image_boxes, index, per_image_size in zip(
                scores, classes, boxes, per_batch_ids, sizes
            ):
                # clip boxes
                per_image_boxes[:, 0] = np.maximum(per_image_boxes[:, 0], 0)
                per_image_boxes[:, 1] = np.maximum(per_image_boxes[:, 1], 0)
                per_image_boxes[:, 2] = np.minimum(per_image_boxes[:, 2], per_image_size[1])
                per_image_boxes[:, 3] = np.maximum(per_image_boxes[:, 3], per_image_size[0])

                # for coco_eval, we need [x_min, y_min, w, h] format pred boxes
                per_image_boxes[:, 2:] -= per_image_boxes[:, :2]

                # 处理每个预测框
                for object_score, object_class, object_box in zip(
                        per_image_scores, per_image_classes, per_image_boxes):
                    object_score = float(object_score)
                    object_class = int(object_class)
                    object_box = object_box.tolist()
                    if object_class == -1:
                        break

                    image_result = {
                        'image_id': test_dataset.image_ids[index],
                        'category_id': test_dataset.coco_label_to_category_id[object_class],
                        'score': object_score,
                        'bbox': object_box
                    }
                    results.append(image_result)
                image_ids.append(test_dataset.image_ids[index])

                print('{}/{}'.format(index, len(test_dataset)), end='\r')
            end = time.time()
            test_loss = losses.avg

            variable_definitions = {
                0: 'IoU=0.50:0.95,area=all,maxDets=100,mAP',
                1: 'IoU=0.50,area=all,maxDets=100,mAP',
                2: 'IoU=0.75,area=all,maxDets=100,mAP',
                3: 'IoU=0.50:0.95,area=small,maxDets=100,mAP',
                4: 'IoU=0.50:0.95,area=medium,maxDets=100,mAP',
                5: 'IoU=0.50:0.95,area=large,maxDets=100,mAP',
                6: 'IoU=0.50:0.95,area=all,maxDets=1,mAR',
                7: 'IoU=0.50:0.95,area=all,maxDets=10,mAR',
                8: 'IoU=0.50:0.95,area=all,maxDets=100,mAR',
                9: 'IoU=0.50:0.95,area=small,maxDets=100,mAR',
                10: 'IoU=0.50:0.95,area=medium,maxDets=100,mAR',
                11: 'IoU=0.50:0.95,area=large,maxDets=100,mAR',
            }

            result_dict = collections.OrderedDict()

            # per image data load time(ms) and inference time(ms)
            per_image_load_time = data_time.avg / batch_size * 1000
            per_image_inference_time = batch_time.avg / batch_size * 1000

            result_dict['test_loss'] = test_loss
            result_dict['per_image_load_time'] = f'{per_image_load_time:.3f}ms'
            result_dict['per_image_inference_time'] = f'{per_image_inference_time:.3f}ms'

            if len(results) == 0:
                for _, value in variable_definitions.items():
                    result_dict[value] = 0
                return result_dict

            # load results in COCO evaluation tool
            coco_true = test_dataset.coco
            coco_pred = coco_true.loadRes(results)

            coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            eval_result = coco_eval.stats

            for i, var in enumerate(eval_result):
                result_dict[variable_definitions[1]] = var * 100

            return result_dict

